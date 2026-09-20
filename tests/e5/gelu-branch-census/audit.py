"""Reconstruct all sixty branch censuses from retained float32 data, without inference."""
from pathlib import Path
import argparse, json, struct
import numpy as np
import psutil
from run import CASES, CORE, MODEL, pin, read, write

def input_hash(inputs):
    import hashlib
    data=bytearray(b'LOKAD-CAMPAIGN-INPUTS-1\0')+struct.pack('<i',len(inputs))
    for key,values in sorted(inputs.items()):
        name=key.encode('utf-8');data+=struct.pack('<i',len(name))+name
        data+=struct.pack('<iiiiq',7,2,1,len(values),len(values))+struct.pack('<'+'q'*len(values),*values)
    return hashlib.sha256(data).hexdigest()

def inspect_capture(base, capture=None):
    data=base/'capture';capture=read(data/'capture.json') if capture is None else capture
    assert capture['schema']==1 and capture['diagnostic']=='uniform-gelu-branch-census'
    assert capture['core_sha256']==CORE and capture['model_sha256']==MODEL
    assert capture['affinity']==4 and capture['vector_width']==8 and capture['settings']=={}
    assert capture['runtime']=='.NET 10.0.12'
    assert capture['probe_sha256']==pin(base/'bin/Census.dll')['sha256']
    assert [c['name'] for c in capture['cases']]==CASES
    assert {p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}==set(capture['files'])|{'capture.json'}
    for name,wanted in capture['files'].items():assert pin(data/name)==wanted,('Captured file hash',name)
    summaries=[];node_names=None
    for case in capture['cases']:
        name=case['name'];folder=data/name;fixture=read(base/'inputs'/(name+'.json'))
        assert case==read(folder/'capture.json') and case['fixture_sha256']==pin(base/'inputs'/(name+'.json'))['sha256']
        assert input_hash(fixture['inputs'])==case['input_sha256']==fixture['input_sha256']
        assert fixture['model_sha256']==MODEL and case['inputs_unchanged']
        actual=np.fromfile(folder/'output.f32',dtype='<f4');reference=np.fromfile(base/'inputs'/fixture['reference_file'],dtype='<f4')
        assert pin(base/'inputs'/fixture['reference_file'])['sha256']==fixture['reference_sha256']
        assert fixture['shape']==case['output_shape'] and actual.size==reference.size==np.prod(fixture['shape'])
        assert np.isfinite(actual).all() and np.isfinite(reference).all()
        error=float(np.max(np.abs(actual.astype('float64')-reference)/np.maximum(1,np.abs(reference.astype('float64')))))
        assert error==case['max_native_error'] and error<=1e-4
        assert [n['index'] for n in case['nodes']]==list(range(12))
        names=[n['name'] for n in case['nodes']];assert len(set(names))==12
        if node_names is None:node_names=names
        assert node_names==names
        totals=dict(values=0,vectors=0,small_lanes=0,all_small=0,all_large=0,mixed=0)
        for node in case['nodes']:
            prefix=f"{node['index']:02d}"
            x=np.fromfile(folder/(prefix+'-x.f32'),dtype='<f4');b=np.fromfile(folder/(prefix+'-bias.f32'),dtype='<f4');y=np.fromfile(folder/(prefix+'-y.f32'),dtype='<f4')
            assert node['shape']==[1,len(fixture['inputs']['input_ids']),1536] and node['bias_shape']==[1536]
            assert len(x)==len(y)==np.prod(node['shape']) and len(b)==1536 and node['product_replay_bitwise']
            assert np.isfinite(x).all() and np.isfinite(b).all() and np.isfinite(y).all()
            biased=np.add(x.reshape(-1,1536),b,dtype='float32')
            scaled=np.multiply(biased,np.float32(.7071067811865476),dtype='float32').reshape(-1,8)
            small=np.abs(scaled)<=np.float32(.921875)
            counts=dict(values=int(x.size),vectors=int(x.size//8),small_lanes=int(small.sum()),all_small=int(small.all(axis=1).sum()),all_large=int((~small).all(axis=1).sum()))
            counts['mixed']=counts['vectors']-counts['all_small']-counts['all_large']
            for key,value in counts.items():assert node[key]==value,(name,node['index'],key);totals[key]+=value
        expected={name+'/output.f32',name+'/capture.json'}|{f'{name}/{i:02d}-{kind}.f32' for i in range(12) for kind in ['x','bias','y']}
        assert {n for n in capture['files'] if n.startswith(name+'/')}==expected
        summaries.append(dict(name=name,max_native_error=error,**totals,
                              all_small_fraction=totals['all_small']/totals['vectors'],all_large_fraction=totals['all_large']/totals['vectors']))
    return summaries

def audit(base):
    frozen=read(base/'frozen.json');process=read(base/'process.json')
    assert process['complete'] and process['exit_code']==0 and not process.get('error')
    for item in [process['child'],process['supervisor']]:
        try:assert psutil.Process(item['pid']).create_time()!=item['birth'],('Still live',item)
        except psutil.NoSuchProcess:pass
    assert frozen['cases']==CASES and frozen['screen']==dict(all_small_fraction=.10,all_large_fraction=.40)
    assert frozen['limits']==dict(rss=6*1024**3,seconds=120,available=1024**3)
    assert process['samples'] and all(0<=s['seconds']<=120 and 0<s['rss']<=6*1024**3 and s['available']>=1024**3 for s in process['samples'])
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,('Frozen file',name)
    assert pin(base/'source/audit.py')==pin(Path(__file__))
    assert read(base/'launch.json')['child']==process['child']
    summaries=inspect_capture(base)
    candidate=any(s['all_small_fraction']>=.10 or s['all_large_fraction']>=.40 for s in summaries[:4])
    return dict(passed=True,cases=summaries,prototype_screen_passed=candidate,
                interpretation='Prioritization screen only; no speedup, optimality or full-model timing conclusion',
                process=pin(base/'process.json'),capture=pin(base/'capture/capture.json'),frozen=pin(base/'frozen.json'),
                auditor=pin(Path(__file__)),numpy=np.__version__,samples=len(process['samples']),
                peak_rss=max(s['rss'] for s in process['samples']),minimum_available=min(s['available'] for s in process['samples']))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    value=audit(args.artifact.resolve());write(args.output,value);print(json.dumps(value,indent=2))
