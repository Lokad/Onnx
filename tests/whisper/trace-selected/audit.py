"""Audit every selected trace array, instrumentation effect and same-input contrast."""
from pathlib import Path
import argparse,collections,importlib.util,json,math,sys
import numpy as np,onnx
from common import ROOT,CORE,KINDS,pin,read,write,verify,schedule,header

helper_path=ROOT/'tests/whisper/input-cross/audit.py'
helper_spec=importlib.util.spec_from_file_location('original_whisper_audit',helper_path)
original=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(original)

def array(path,record,description):
    assert record['index']==description['index'] and record['name']==description['name'] and record['shape']==description['shape']
    count=math.prod(description['shape']);assert record['values']==count
    assert pin(path)==dict(bytes=count*4,sha256=record['sha256'])
    value=np.fromfile(path,dtype='<f4').reshape(description['shape']);assert np.isfinite(value).all()
    return value

def instrumentation(actual,reference,record):
    value=original.metrics(actual.astype(np.float64)-reference.astype(np.float64),np.maximum(1,np.abs(reference.astype(np.float64))))
    assert record['baseline_sha256']==original.raw(reference) and record['final_sha256']==original.raw(actual)
    assert record['bitwise'] is (original.raw(reference)==original.raw(actual))
    assert record['failed_values']==value['failed_values'] and record['max_scaled']==value['max_scaled']
    return dict(bitwise=record['bitwise'],**value)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    spec=read(base/'manifest.json');verify(spec);jobs=schedule(spec);assert spec['schedule']==jobs and (spec['workers'],spec['calls'],spec['arrays'])==(8,16,656)
    state=read(base/'campaign.json');assert state['complete'] is True and state['code']==0 and not state.get('error') and original.absent(state['supervisor'])
    assert state['manifest_sha256']==pin(base/'manifest.json')['sha256'] and [r['job'] for r in state['runs']]==jobs
    assert all(a['ended']<=b['started'] for a,b in zip(state['runs'],state['runs'][1:]))
    preflight=[json.loads(line) for line in (base/'preflight.jsonl').read_text().splitlines()];seen_preflight=[]
    assert all(a['time']<=b['time'] for a,b in zip(preflight,preflight[1:]))
    arrays={};telemetry=[];effects=[];censuses={};first={};bytes_saved=0
    for run,job in zip(state['runs'],jobs):
        observed=[r for r in preflight if r['job']==job['id']];assert observed
        assert observed[-1]['available']==run['preflight_available'] and observed[-1]['time']<=run['started']
        assert all(r['available']<spec['limits']['preflight_available'] for r in observed[:-1])
        assert observed[-1]['time']-observed[0]['time']<=spec['preflight_wait_seconds']+6
        seen_preflight += [job['id']]*len(observed)
        samples=[json.loads(line) for line in (base/'process'/job['id']/'samples.jsonl').read_text().splitlines()]
        assert run['supervisor']==state['supervisor'];resource=original.resources(run,samples,spec)
        assert all(original.absent(item) for item in resource['births']);telemetry.append(dict(job=job,**resource))
        folder=base/'outputs'/job['id'];result=read(folder/'result.json');header(result,job,pin(base/'manifest.json')['sha256'])
        engine=job['engine'];index=job['request'];item=spec['requests'][index];expected_files={'result.json'}
        assert sum(result['node_census'].values())==result['optimized_nodes']>0
        if engine in censuses:assert result['node_census']==censuses[engine]
        else:censuses[engine]=result['node_census']
        if engine=='managed':
            assert result['core_sha256']==CORE and result['probe_sha256']==pin(base/'bin/WhisperSelectedTrace.dll')['sha256']
            assert result['runtime']=='10.0.12' and result['affinity']==4 and result['processor_count']==1
            assert result['native_loaded'] is False and result['packed_weight_bytes']==256*1024**2
        else:
            assert result['native_runtime']==spec['native_runtime'] and result['affinity']==[2] and result['threads']==1
            assert result['sequential'] is True and result['all_optimizations'] is True and result['spinning'] is False
            assert result['modules'] and all(spec['native_runtime']['files'].get(k)==v for k,v in result['modules'].items())
            assert result['serialization']==spec['native_serialization'] and set(result['optimized_files'])=={'optimized.onnx','optimized.weights'}
            for name,want in result['optimized_files'].items():assert pin(folder/name)==want;expected_files.add(name)
            optimized=onnx.load(folder/'optimized.onnx',load_external_data=False)
            assert dict(collections.Counter((n.domain+'::' if n.domain else '')+n.op_type for n in optimized.graph.node))==result['node_census']
            assert [v.name for v in optimized.graph.output]==[v['name'] for v in spec['outputs']]
        held={}
        for call in result['records']:
            kind=call['kind'];assert call['request']==index and call['name']==item['name']
            feature=item['managed_features' if kind in ['MM','NM'] else 'native_features'];original.source(feature)
            assert call['input_sha256']==feature['raw_sha256']
            for number,(record,description) in enumerate(zip(call['outputs'],spec['outputs'])):
                filename=f'{kind}-{number:02}.f32';assert record['file']==filename;path=folder/filename
                value=array(path,record,dict(index=number,**description));held[kind+':'+str(number)]=record['sha256'];bytes_saved+=value.nbytes
                expected_files.add(filename);arrays[(index,kind,number)]=path
                if index==0:first[kind,number]=record['sha256']
                if index==3:assert first[kind,number]==record['sha256'],'Repeated selected trace changed'
                if number==40:
                    reference=original.source(item['baselines'][kind]);effects.append(dict(request=index,name=item['name'],kind=kind,**instrumentation(value,reference,call['instrumentation'])))
        assert result['held_outputs']==held and {p.name for p in folder.iterdir()}==expected_files
    assert [r['job'] for r in preflight]==seen_preflight and bytes_saved==spec['output_payload_bytes']==5283840000 and len(arrays)==656
    for name in ['outputs','process']:assert {p.name for p in (base/name).iterdir()}=={j['id'] for j in jobs}
    contrasts=[]
    for index,item in enumerate(spec['requests']):
        for features,managed,native in [('managed','MM','NM'),('native','MN','NN')]:
            nodes=[]
            for number,description in enumerate(spec['outputs']):
                a=np.fromfile(arrays[index,managed,number],dtype='<f4').astype(np.float64)
                b=np.fromfile(arrays[index,native,number],dtype='<f4').astype(np.float64)
                nodes.append(dict(index=number,**description,**original.metrics(a-b,np.maximum(1,np.abs(b)))))
            contrasts.append(dict(request=index,name=item['name'],features=features,outputs=nodes))
    files={p.relative_to(base).as_posix():pin(p) for name in ['outputs','process'] for p in sorted((base/name).rglob('*')) if p.is_file()}
    for name in ['campaign.json','preflight.jsonl']:files[name]=pin(base/name)
    verify(spec)
    value=dict(passed=True,manifest=pin(base/'manifest.json'),arrays=656,saved_array_bytes=bytes_saved,workers=8,calls=16,
        instrumentation=effects,contrasts=contrasts,resources=telemetry,censuses=censuses,files=files,
        scope='Selected-case localization; native traced array denominators; original full-corpus acceptance unchanged')
    write(args.output,value);print(json.dumps(dict(arrays=656,bytes=bytes_saved,instrumentation=effects,censuses=censuses),indent=2))

if __name__=='__main__':main()
