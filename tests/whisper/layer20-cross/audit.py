"""Audit all cut arrays, extraction effects, and common-input decompositions."""
from pathlib import Path
import argparse,collections,importlib.util,json,math
import numpy as np,onnx
from common import ROOT,CORE,KINDS,pin,read,write,verify,schedule,header

loader=importlib.util.spec_from_file_location('prior_input_cross',ROOT/'tests/whisper/input-cross/audit.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
TERMS=['diagonal_MM-NN','engine_MM-NM','engine_MN-NN','input_MM-MN','input_NM-NN','interaction']

def array(path,record,description):
    assert record['index']==description['index'] and record['name']==description['name'] and record['shape']==description['shape']
    count=math.prod(description['shape']);assert record['values']==count and pin(path)==dict(bytes=count*4,sha256=record['sha256'])
    value=np.fromfile(path,dtype='<f4').reshape(description['shape']);assert np.isfinite(value).all();return value

def metrics(delta,denominator):
    value=original.metrics(delta,denominator);scaled=np.abs(delta)/denominator;index=int(np.argmax(scaled))
    return dict(**value,rms=math.sqrt(value['sum_squares']/value['values']),maximum_index=list(map(int,np.unravel_index(index,delta.shape))))

def instrumentation(actual,reference,record):
    delta=actual.astype(np.float64)-reference.astype(np.float64);denominator=np.maximum(1,np.abs(reference.astype(np.float64)));value=metrics(delta,denominator)
    assert record['baseline_sha256']==original.raw(reference) and record['final_sha256']==original.raw(actual)
    assert record['bitwise'] is (original.raw(reference)==original.raw(actual))
    assert record['failed_values']==value['failed_values'] and record['max_scaled']==value['max_scaled']
    return dict(bitwise=record['bitwise'],**value)

def input_record(item,kind):
    assert kind in ['MM','MN','NM','NN'];return item['managed_input' if kind[1]=='M' else 'native_input']

def decompose(cells):
    assert set(cells)=={'MM','MN','NM','NN'}
    mm,mn,nm,nn=[np.asarray(cells[k],dtype=np.float64) for k in ['MM','MN','NM','NN']]
    assert all(v.shape==mm.shape and np.isfinite(v).all() for v in [mn,nm,nn]) and np.isfinite(mm).all()
    deltas=[mm-nn,mm-nm,mn-nn,mm-mn,nm-nn,(mm-mn)-(nm-nn)];denominator=np.maximum(1,np.abs(nn))
    residuals=[float(np.abs(deltas[0]-deltas[1]-deltas[4]).max()),float(np.abs(deltas[0]-deltas[3]-deltas[2]).max())]
    assert max(residuals)<=1e-12
    terms={key:metrics(delta,denominator) for key,delta in zip(TERMS,deltas)}
    pairs={key:metrics(delta,np.maximum(1,np.abs(reference))) for key,delta,reference in [('managed_incoming',mm-nm,nm),('native_incoming',mn-nn,nn)]}
    return dict(common_reference='native cut on native incoming array (NN)',terms=terms,closure_max=residuals,pairwise_native_reference=pairs)

def audit(base):
    spec=read(base/'manifest.json');verify(spec);jobs=schedule(spec);assert spec['schedule']==jobs
    state=read(base/'campaign.json');assert state['complete'] is True and state['code']==0 and not state.get('error') and original.absent(state['supervisor'])
    assert state['manifest_sha256']==pin(base/'manifest.json')['sha256'] and [r['job'] for r in state['runs']]==jobs
    assert all(a['ended']<=b['started'] for a,b in zip(state['runs'],state['runs'][1:]))
    preflight=[json.loads(line) for line in (base/'preflight.jsonl').read_text().splitlines()];seen=[]
    assert all(a['time']<=b['time'] for a,b in zip(preflight,preflight[1:]))
    paths={};telemetry=[];effects=[];censuses={};first={};bytes_saved=0
    for run,job in zip(state['runs'],jobs):
        observed=[r for r in preflight if r['job']==job['id']];assert observed
        assert observed[-1]['available']==run['preflight_available'] and observed[-1]['time']<=run['started']
        assert all(r['available']<spec['limits']['preflight_available'] for r in observed[:-1])
        assert observed[-1]['time']-observed[0]['time']<=spec['preflight_wait_seconds']+6;seen += [job['id']]*len(observed)
        samples=[json.loads(line) for line in (base/'process'/job['id']/'samples.jsonl').read_text().splitlines()]
        assert run['supervisor']==state['supervisor'];resource=original.resources(run,samples,spec)
        assert all(original.absent(item) for item in resource['births']);telemetry.append(dict(job=job,seconds=run['seconds'],**resource))
        folder=base/'outputs'/job['id'];result=read(folder/'result.json');header(result,job,pin(base/'manifest.json')['sha256'])
        engine=job['engine'];index=job['request'];item=spec['requests'][index];expected={'result.json'}
        assert sum(result['node_census'].values())==result['optimized_nodes']>0
        if engine in censuses:assert result['node_census']==censuses[engine]
        else:censuses[engine]=result['node_census']
        if engine=='managed':
            assert result['core_sha256']==CORE and result['probe_sha256']==pin(base/'bin/WhisperLayer20Cross.dll')['sha256']
            assert result['runtime']=='10.0.12' and result['affinity']==4 and result['processor_count']==1
            assert result['native_loaded'] is False and result['packed_weight_bytes']==256*1024**2
        else:
            assert result['native_runtime']==spec['native_runtime'] and result['affinity']==[2] and result['threads']==1
            assert result['sequential'] is True and result['all_optimizations'] is True and result['spinning'] is False
            assert result['modules'] and all(spec['native_runtime']['files'].get(k)==v for k,v in result['modules'].items())
            assert result['serialization']==spec['native_serialization'] and set(result['optimized_files'])=={'optimized.onnx','optimized.weights'}
            for name,want in result['optimized_files'].items():assert pin(folder/name)==want;expected.add(name)
            optimized=onnx.load(folder/'optimized.onnx',load_external_data=False)
            assert dict(collections.Counter((n.domain+'::' if n.domain else '')+n.op_type for n in optimized.graph.node))==result['node_census']
            assert [v.name for v in optimized.graph.output]==[v['name'] for v in spec['outputs']]
        held={}
        for call in result['records']:
            kind=call['kind'];assert call['request']==index and call['name']==item['name']
            source=input_record(item,kind);original.source(source);assert call['input_sha256']==source['raw_sha256']
            for number,(record,description) in enumerate(zip(call['outputs'],spec['outputs'])):
                filename=f'{kind}-{number:02}.f32';assert record['file']==filename;path=folder/filename
                value=array(path,record,dict(index=number,**description));held[kind+':'+str(number)]=record['sha256'];bytes_saved+=value.nbytes
                expected.add(filename);paths[index,kind,number]=path
                repeat_key=(item['features'],kind,number)
                if item['selected_request']==0:first[repeat_key]=record['sha256']
                if item['selected_request']==3:assert first[repeat_key]==record['sha256'],'Repeated cut output changed'
                if number==11:
                    reference=original.source(item['baselines'][kind]);effects.append(dict(request=index,name=item['name'],features=item['features'],kind=kind,
                        extraction_bridge=kind[0]==kind[1],**instrumentation(value,reference,call['instrumentation'])))
        assert result['held_outputs']==held and {p.name for p in folder.iterdir()}==expected
    assert [r['job'] for r in preflight]==seen and bytes_saved==spec['output_payload_bytes']==4423680000 and len(paths)==384
    for name in ['outputs','process']:assert {p.name for p in (base/name).iterdir()}=={j['id'] for j in jobs}
    contrasts=[]
    for index,item in enumerate(spec['requests']):
        nodes=[]
        for number,description in enumerate(spec['outputs']):
            cells={kind:np.fromfile(paths[index,kind,number],dtype='<f4').reshape(description['shape']) for kind in ['MM','MN','NM','NN']}
            nodes.append(dict(index=number,**description,**decompose(cells)))
        contrasts.append(dict(request=index,name=item['name'],selected_request=item['selected_request'],features=item['features'],outputs=nodes))
    files={p.relative_to(base).as_posix():pin(p) for name in ['outputs','process'] for p in sorted((base/name).rglob('*')) if p.is_file()}
    for name in ['campaign.json','preflight.jsonl']:files[name]=pin(base/name)
    verify(spec);bridges=[r for r in effects if r['extraction_bridge']];assert len(bridges)==16
    return dict(passed=True,extraction_bridges_passed=all(r['failed_values']==0 for r in bridges),manifest=pin(base/'manifest.json'),arrays=384,saved_array_bytes=bytes_saved,
        workers=16,calls=32,instrumentation=effects,contrasts=contrasts,resources=telemetry,censuses=censuses,files=files,
        scope='Selected-case layer20 localization; complete intermediate data retained; original full-corpus numerical acceptance unchanged')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.artifact.resolve());write(args.output,result)
    print(json.dumps(dict(arrays=result['arrays'],extraction_bridges_passed=result['extraction_bridges_passed'],instrumentation=result['instrumentation'],censuses=result['censuses']),indent=2))
