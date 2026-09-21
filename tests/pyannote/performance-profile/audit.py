"""Independently check attribution outputs, application results and raw clocks."""
import collections
import json
from pathlib import Path
import sys
import numpy as np
from run import ROOT, BASE, MANIFEST, pin
import psutil


def absent(identity):
    try:
        return psutil.Process(identity['pid']).create_time() != identity['birth']
    except psutil.NoSuchProcess:
        return True


def main():
    assert not (BASE/'audit.json').exists()
    frozen=json.loads((BASE/'frozen.json').read_text())
    for name, expected in frozen['files'].items():
        assert pin(ROOT/name)==expected,name
    state=json.loads((BASE/'state.json').read_text())
    assert state['complete'] and state['code']==0 and state['passed']
    assert absent(state['supervisor']) and absent(state['worker'])
    result=json.loads((BASE/'output/result.json').read_text())
    spec=json.loads(MANIFEST.read_text())
    assert result['passed'] and result['inputs_and_held_outputs_unchanged']
    assert result['manifest_sha256']==pin(MANIFEST)['sha256'] and result['runtime']=='10.0.12'
    crops=[c['name'] for c in spec['cases'] if c['samples']==160000]
    assert [(r['name'],r['model'],r['pass']) for r in result['rows']]==[(c,m,p) for c in crops for m in ('segmentation','embedding') for p in range(3)]
    profiles=[];arrays={};groups={}
    for row in result['rows']:
        for kind in ('input','output'):
            info=row[kind];path=BASE/'output'/info['file'];actual=np.fromfile(path,dtype='<f4')
            assert pin(path)['sha256']==info['sha256'] and actual.size==info['values']==int(np.prod(info['shape']))
            assert np.isfinite(actual).all();arrays[info['file']]=info
        group=(row['name'],row['model']);groups.setdefault(group,[]).append(row)
        assert row['seconds']==(row['end_ticks']-row['start_ticks'])/row['frequency']>0
        assert len(row['nodes'])==((108 if row['model']=='segmentation' else 75) if row['pass']==2 else 0)
        if row['pass']==2:
            by=collections.defaultdict(float);last=row['start_ticks']
            assert len({n['id'] for n in row['nodes']})==len(row['nodes'])
            for n in row['nodes']:
                assert last<=n['start_ticks']<=n['end_ticks']<=row['end_ticks']
                assert n['seconds']==(n['end_ticks']-n['start_ticks'])/row['frequency']
                by[n['op']]+=n['seconds'];last=n['end_ticks']
            profiles.append(dict(name=row['name'],model=row['model'],seconds=row['seconds'],operators=dict(by),
                outside_nodes_seconds=row['seconds']-sum(by.values())))
    for rows in groups.values():
        assert len({r['input']['sha256'] for r in rows})==len({r['output']['sha256'] for r in rows})==1
    assert len(arrays)==24 and {p.name for p in (BASE/'output').iterdir()}==set(arrays)|{'result.json'}
    assert [a['name'] for a in result['applications']]==[c['name'] for c in spec['cases']]
    max_error=0.
    for row,case in zip(result['applications'],spec['cases'],strict=True):
        a=row['result'];e=case['expected']
        assert a['Status']==0 and e['status']=='Completed' and a['Windows']==e['windows'] and a['AudioDuration']==e['audio_seconds']
        for managed,native in [('Intervals','intervals'),('ExclusiveIntervals','exclusive_intervals')]:
            assert len(a[managed])==len(e[native])
            for observed,wanted in zip(a[managed],e[native],strict=True):
                assert observed['Speaker']==wanted[2]
                assert abs(observed['Start']-wanted[0])<=1e-12 and abs(observed['End']-wanted[1])<=1e-12
        assert len(a['Speakers'])==len(e['speakers']);maximum=0.
        for observed,wanted in zip(a['Speakers'],e['speakers'],strict=True):
            assert observed['Speaker']==wanted['speaker'] and observed['HasEmbedding']==wanted['has_embedding']
            left=np.array(observed['Centroid']);right=np.array(wanted['centroid'])
            assert left.shape==right.shape==(256,) and np.isfinite(left).all()
            maximum=max(maximum,float(np.max(np.abs(left-right)/np.maximum(1.,np.abs(right)))))
        assert maximum<=1e-4 and maximum==row['maximum_centroid_error'];max_error=max(max_error,maximum)
    samples=[json.loads(s) for s in (BASE/'samples.jsonl').read_text().splitlines()]
    assert state['preflight']['available']>=10*1024**3 and state['preflight']['disk']>=20*1024**3
    for s in samples:
        assert s['seconds']<900 and s['rss']<8*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['affinity']==[2]
    output=dict(passed=True,profiles=profiles,output_arrays=18,input_arrays=6,output_values=sum(r['output']['values'] for r in result['rows']),
        public_calls=4,maximum_centroid_error=max_error,samples=len(samples),peak_rss=max(s['rss'] for s in samples),
        identities=[state['supervisor'],state['worker']],limitation='Local attribution only; no matched AMD performance result')
    (BASE/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
    files=dict(frozen['files'])
    for path in BASE.rglob('*'):
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    files[str(Path(__file__).resolve().relative_to(ROOT))]=pin(Path(__file__).resolve())
    (BASE/'closed.json').write_text(json.dumps(dict(passed=True,files=files,identities=output['identities']),indent=2)+'\n')
    print(json.dumps(dict(passed=True,profiles=profiles,arrays=len(arrays),output_values=output['output_values'],
        samples=len(samples),peak_rss=output['peak_rss'],files=len(files),closure=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
