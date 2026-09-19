"""Recompute every saved public WeSpeaker vector and status independently."""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np

def audit(reference, result):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    m = json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    r = json.loads(result.read_text(encoding='utf-8'))
    pins = json.loads(Path(__file__).with_name('pins.json').read_text(encoding='utf-8'))
    assert r['manifest_sha256'] == sha(reference/'manifest.json') and m['pins'] == pins
    for key, file in [('pins_lf_sha256',Path(__file__).with_name('pins.json')),
                      ('generator_lf_sha256',Path(__file__).with_name('generate_reference.py')),
                      ('prepare_lf_sha256',Path(__file__).resolve().parents[3]/'eng/prepare-wespeaker-projection.py')]:
        assert m[key] == hashlib.sha256(file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert sha(reference/'projection.onnx') == m['projection']['output_sha256']
    arrays = {}
    for file, meta in m['files'].items():
        assert Path(file).name == file
        path = reference/file
        assert sha(path) == meta['sha256'] and path.stat().st_size == meta['bytes']
        a = np.load(path,allow_pickle=False)
        assert a.dtype == np.dtype('float32') and meta['dtype']=='float32' and list(a.shape)==meta['shape'] and np.isfinite(a).all()
        arrays[file] = a
    coverage = {a+'-'+b for a in pins['recordings'] for b in pins['masks']}
    expected = {}; used = set()
    for c in m['cases']:
        coverage.remove(c['name']); assert c['name']==c['recording']+'-'+c['mask_kind']
        pcm = arrays[c['pcm']]; used.add(c['pcm'])
        assert pcm.ndim==1 and 400<=len(pcm)<=480000 and (np.abs(pcm)<=1).all()
        frames = (1+(len(pcm)-400)//160+7)//8
        assert (c['mask'] is None)==(c['mask_kind']=='none')
        mask = None if c['mask'] is None else arrays[c['mask']]
        if mask is not None:
            used.add(c['mask']); assert mask.ndim==1 and 1<=len(mask)<=480000 and ((mask>=0)&(mask<=1)).all()
        weights = np.ones(frames) if mask is None else mask[np.arange(frames)*len(mask)//frames]
        positive = int(np.count_nonzero(weights>0)); status = 'Completed' if positive>=2 else 'InsufficientFrames'
        assert (c['frames'],c['positive'],c['status'])==(frames,positive,status)
        assert (c['vector'] is not None)==(positive>=2)
        if c['vector'] is not None: used.add(c['vector']); assert arrays[c['vector']].shape==(1,256)
        for repeat in range(pins['repeats']): expected[c['name'],repeat]=c
    assert not coverage and used==set(arrays)
    count=bad_count=vector_count=0; maximum=0.; rows=[]; bits={}; files=set()
    for row in r['reports']:
        c=expected.pop((row['name'],row['repeat']))
        assert (row['status'],row['frames'],row['positive'])==(c['status'],c['frames'],c['positive'])
        if c['vector'] is None:
            assert row['file'] is None and row['values']==0 and row['bad']==0 and row['error']==0
            assert row['sha256']==hashlib.sha256(b'').hexdigest()
            continue
        assert Path(row['file']).name==row['file']; files.add(row['file'])
        p=Path(str(result)+'.arrays')/row['file']; assert sha(p)==row['sha256']
        a=np.fromfile(p,np.float32); b=arrays[c['vector']].reshape(-1)
        assert a.shape==b.shape==(256,) and row['values']==256 and np.isfinite(a).all()
        errors=np.abs(a.astype(np.float64)-b)/np.maximum(1,np.abs(b.astype(np.float64)))
        error=float(errors.max()); bad=int(np.count_nonzero(errors>pins['tolerance']))
        assert math.isclose(error,row['error'],rel_tol=1e-15,abs_tol=0) and bad==row['bad']
        if row['name'] in bits: assert bits[row['name']]==row['sha256']
        bits[row['name']]=row['sha256']; count+=a.size; vector_count+=1; bad_count+=bad; maximum=max(maximum,error)
        rows.append(dict(name=row['name'],repeat=row['repeat'],maximum=error,bad=bad))
    assert not expected and files=={p.name for p in Path(str(result)+'.arrays').iterdir()}
    assert count==r['values'] and math.isclose(maximum,r['maximum'],rel_tol=1e-15,abs_tol=0) and r['passed']==(bad_count==0)
    assert r['refusals']==['cancelled','rate','short','long','pcm','mask'] and r['concurrent_requests']==3
    return dict(passed=bad_count==0,cases=len(m['cases']),requests=len(r['reports']),vectors=vector_count,values=count,
        maximum=maximum,bad=bad_count,result_sha256=sha(result),rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); answer=audit(args.reference,args.result)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(answer,stream,indent=2)
    print(json.dumps({k:v for k,v in answer.items() if k!='rows'},indent=2));raise SystemExit(0 if answer['passed'] else 1)
