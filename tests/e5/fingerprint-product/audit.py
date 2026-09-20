"""Independently verify complete arrays, frozen identities, TRX and resource telemetry."""
from pathlib import Path
import argparse,hashlib,json,xml.etree.ElementTree as ET
import numpy as np

CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def arrays(base,root,mode,enabled,meta):
    output=base/f'{mode}-{enabled}/output';result=read(output/'result.json')
    assert result['passed'] is True and result['enabled'] is bool(enabled) and result['mode']==mode
    assert result['core_sha256']==meta['core']['sha256'] and result['probe_sha256']==meta['files'][meta['paths']['replay']+'/Replay.dll']['sha256']
    assert result['flags']=={'LOKAD_ONNX_FINGERPRINT_STRINGS':str(enabled)}
    assert result['inputs_unchanged'] is True and result['held_outputs_unchanged'] is True
    if mode=='e5':
        reference=base.parent/'inputs';expected=[]
        for case in CASES:
            fixture=read(reference/(case+'.json'))
            for policy in ['default','memory']:
                for context in ['facade','context']:
                    for step in range(3):expected.append((case,policy+'-'+context,step,'last_hidden_state',fixture['reference_file']))
        assert len(result['graphs'])==80
    else:
        reference=root/'artifacts/shared-regression-20260918/reference';expected=[]
        manifest=read(reference/'manifest.json')
        for model in manifest['models']:
            for scenario in model['scenarios']:
                for step,values in enumerate(scenario['steps']):
                    for item in values['outputs']:expected.append((model['key'],scenario['name'],step,item['name'],item['file']))
        assert len(result['graphs'])==len(manifest['models'])+sum(len(s['steps']) for m in manifest['models'] for s in m['scenarios'])
    assert [(r['model'],r['scenario'],r['step'],r['name'],r['reference_file']) for r in result['rows']]==expected
    for graph in result['graphs']:assert (graph['entries']>0 if enabled else graph['entries']==0) and isinstance(graph['fingerprint'],int)
    maximum=0.;count=0
    for i,row in enumerate(result['rows']):
        assert row['file']==str(i)+'.f32' and row['sha256']==pin(output/row['file'])['sha256']
        assert row['reference_sha256']==pin(reference/row['reference_file'])['sha256']
        want=np.fromfile(reference/row['reference_file'],dtype='<f4') if mode=='e5' else np.load(reference/row['reference_file'])
        actual=np.fromfile(output/row['file'],dtype='<f4')
        assert want.size==actual.size==row['values']==int(np.prod(row['shape']))
        if mode=='shared':assert list(want.shape)==row['shape']
        else:assert read(reference/(row['model']+'.json'))['shape']==row['shape']
        assert np.isfinite(actual).all() and np.isfinite(want).all()
        error=np.abs(actual.astype(np.float64)-want.reshape(-1).astype(np.float64))/np.maximum(1,np.abs(want.reshape(-1).astype(np.float64)))
        observed=float(error.max(initial=0))
        assert observed<=1e-4 and row['failed_values']==0 and abs(observed-row['max_scaled_error'])<=1e-15
        maximum=max(maximum,observed);count+=actual.size
    assert {p.name for p in output.iterdir()}=={'result.json'}|{str(i)+'.f32' for i in range(len(expected))}
    return result,dict(arrays=len(expected),values=count,max_scaled_error=maximum)

def audit(payload,root,label):
    meta=read(payload/'frozen.json');base=payload/('result-'+label);state=read(base/'identity.json')
    for name,wanted in meta['files'].items():assert pin(payload/name)==wanted,name
    for name,wanted in meta['assets'].items():assert pin(root/name)==wanted,name
    assert state['complete'] is True and state['code']==0 and state['frozen']==pin(payload/'frozen.json')
    assert state['limits']==meta['limits'] and [r['job'] for r in state['runs']]==meta['jobs']
    ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'};tests={};resources=[];births=[state['supervisor']]
    for run in state['runs']:
        folder=base/run['job']['name'];assert run['code']==0 and 0<run['seconds']<state['limits']['seconds']
        assert run['members'][str(run['child']['pid'])]==run['child']['birth']
        samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        assert len(samples)==run['samples']>0
        peak=0;previous=-1.;minimum=2**64
        for sample in samples:
            assert previous<=sample['seconds']<state['limits']['seconds'];previous=sample['seconds']
            assert sample['available']>=state['limits']['available'];minimum=min(minimum,sample['available'])
            for item in sample['members']:
                assert run['members'][str(item['pid'])]==item['birth'] and item['affinity']==[2] and item['rss']>=0
            rss=sum(item['rss'] for item in sample['members']);assert rss<state['limits']['rss'];peak=max(peak,rss)
        assert peak==run['peak_rss']
        births.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
        resources.append(dict(job=run['job']['name'],seconds=run['seconds'],samples=len(samples),peak_rss=peak,minimum_available=minimum))
        if run['job']['kind'] in ['backend','tensors']:
            doc=ET.parse(folder/'tests.trx');counters=doc.find('.//t:Counters',ns).attrib
            total=3129 if run['job']['kind']=='backend' else 342
            assert int(counters['total'])==total and int(counters['failed'])==0
            assert int(counters['passed'])>=(3036 if run['job']['kind']=='backend' else 342)
            results=doc.findall('.//t:UnitTestResult',ns);assert len(results)==total
            assert sum(r.get('outcome')=='Passed' for r in results)==int(counters['passed'])
            if run['job']['kind']=='backend':
                focused=[r for r in results if 'FingerprintStringCacheTests.' in r.get('testName','')]
                assert len(focused)==9 and all(r.get('outcome')=='Passed' for r in focused)
            tests[run['job']['name']]=counters
    replay={}
    for mode in ['e5','shared']:
        off,off_summary=arrays(base,root,mode,0,meta);on,on_summary=arrays(base,root,mode,1,meta)
        assert [r['sha256'] for r in off['rows']]==[r['sha256'] for r in on['rows']]
        assert [(r['name'],r['fingerprint']) for r in off['graphs']]==[(r['name'],r['fingerprint']) for r in on['graphs']]
        replay[mode]=dict(off=off_summary,on=on_summary,bit_identical=True)
    return dict(passed=True,source_revision=meta['source_revision'],core=meta['core'],frozen=pin(payload/'frozen.json'),
                identity=pin(base/'identity.json'),tests=tests,replay=replay,resources=resources,births=births)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();value=audit(a.payload.resolve(),a.assets.resolve(),a.label)
    with a.output.open('x') as stream:json.dump(value,stream,indent=2)
    print(json.dumps({k:value[k] for k in ['passed','tests','replay','resources']}))
