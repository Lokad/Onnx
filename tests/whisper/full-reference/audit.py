"""Audit each value of both references, then retained FP32 error against both."""
import argparse,math
import onnx
from common import *

def resources(run,samples,supervisor):
    assert run['complete'] is True and run['code']==0 and not run.get('error')
    assert 0<run['seconds']<LIMITS['seconds'] and run['started']<=run['ended']
    assert run['preflight_available']>=LIMITS['preflight_available'] and run['preflight_disk']>=LIMITS['disk']
    assert run['worker']['birth']>=supervisor['birth'] and absent(run['worker'])
    assert len(samples)==run['samples']>0
    for row in samples:
        assert row['pid']==run['worker']['pid'] and row['birth']==run['worker']['birth']
        assert row['rss']<LIMITS['rss'] and row['available']>=LIMITS['available'] and row['affinity']==[2] and 0<=row['seconds']<=run['seconds']
    gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[run['seconds']-samples[-1]['seconds']]
    assert min(gaps)>=0 and max(gaps)<10
    return dict(job=run['job']['id'],samples=len(samples),seconds=run['seconds'],peak_rss=max(r['rss'] for r in samples),min_available=min(r['available'] for r in samples))

def audit(base,phase):
    spec=read(base/'manifest.json');assert spec['limits']==LIMITS and spec['reference_limit']==1e-9 and spec['original_limit']==1e-4
    verify(spec['files']);manifest=pin(base/'manifest.json');runs=[];states=[]
    for name in (['bridge'] if phase=='bridge' else ['bridge','remaining']):
        state=read(base/(name+'.json'));assert state['complete'] and state['code']==0 and not state.get('error') and state['manifest']==manifest and absent(state['supervisor'])
        runs+=state['runs'];states.append(state)
    jobs=spec['jobs'][:2] if phase=='bridge' else spec['jobs'];assert [r['job'] for r in runs]==jobs
    assert all(a['ended']<=b['started'] for a,b in zip(runs,runs[1:]));summaries=[];results={};bindings={}
    nodes=onnx.load(ROOT/spec['trace_model'],load_external_data=False).graph.node;stages=read(ROOT/spec['promoted']/'stages.json')
    numerical={str(Path(n).resolve()).casefold():v for n,v in spec['numerical_files'].items()}
    for state in states:
        for run in state['runs']:
            job=run['job'];process=base/'process'/job['id'];samples=[json.loads(v) for v in (process/'samples.jsonl').read_text().splitlines()]
            summaries.append(resources(run,samples,state['supervisor']));folder=base/'outputs'/job['id'];result=read(folder/'result.json')
            assert result['complete'] and result['job']==job and result['manifest']==manifest and result['input_unchanged']
            assert result['input_sha256']==job['input']['raw_sha256'];source(job['input'])
            rt=result['runtime'];assert rt['pid']==run['worker']['pid'] and rt['birth']==run['worker']['birth'] and rt['affinity']==[2] and rt['blas_threads']==1
            assert rt['native_loaded'] is (job['engine']=='ort')
            for path,wanted in rt['loaded'].items():assert numerical[str(Path(path).resolve()).casefold()]==wanted and pin(path)==wanted,path
            if job['engine']=='numpy':
                assert [(r['index'],r['name'],r['op']) for r in result['records']]==[(i,n.name,n.op_type) for i,n in enumerate(nodes)]
                assert result['settings']==dict(engine='numpy',erf='scipy.special.erf double',numpy='2.2.4')
            else:
                assert len(result['records'])==len(stages)==69
                for record,stage in zip(result['records'],stages):
                    assert record['index']==stage['index'] and record['kind']==stage['kind'] and [r['name'] for r in record['outputs']]==stage['outputs']
                    expected=dict(erf='math.erf',dtype='float64') if stage['kind']=='math_erf' else dict(intra=1,inter=1,execution='ExecutionMode.ORT_SEQUENTIAL',optimizations='GraphOptimizationLevel.ORT_DISABLE_ALL',providers=['CPUExecutionProvider'],intra_spinning='0',inter_spinning='0')
                    assert record['settings']==expected
            assert len(result['outputs'])==41
            expected={'result.json'}
            for index,(desc,row) in enumerate(zip(spec['outputs'],result['outputs'])):
                assert row['index']==index and row['name']==desc['name'] and row['shape']==desc['shape'] and row['file']==f'{index:02}.f64'
                assert row['pin']['bytes']==math.prod(desc['shape'])*8 and pin(folder/row['file'])==row['pin']
                expected.add(row['file'])
            assert {p.name for p in folder.iterdir()}==expected
            results[job['id']]=result
            for directory in [folder,process]:
                for path in directory.iterdir():bindings[path.relative_to(base).as_posix()]=pin(path)
    comparisons=[];fp32=[];first={};all_pass=True;total_bytes=0
    for index in range(0,len(jobs),2):
        left,right=jobs[index:index+2];assert left['engine']=='numpy' and right['engine']=='ort' and left['request']==right['request'] and left['features']==right['features']
        for output in range(41):
            values=[]
            for job in [left,right]:
                row=results[job['id']]['outputs'][output];value=np.fromfile(base/'outputs'/job['id']/row['file'],dtype='<f8').reshape(row['shape'])
                assert np.isfinite(value).all();values.append(value);total_bytes+=value.nbytes
                key=(job['engine'],job['features'],output)
                if job['request']==0:first[key]=row['pin']
                if job['request']==20:assert first[key]==row['pin'],'Reference repeat differs'
            observation=metric(values[0],values[1],1e-9);all_pass&=observation['failed_values']==0
            comparisons.append(dict(request=left['request'],features=left['features'],output=output,name=spec['outputs'][output]['name'],**observation))
            if output==40:
                item=spec['requests'][left['request']];suffix='M' if left['features']=='managed' else 'N'
                for engine,prefix in [('managed','M'),('native','N')]:
                    actual=source(item['baselines'][prefix+suffix])
                    for ref_name,reference in zip(['numpy','ort'],values):
                        fp32.append(dict(request=left['request'],name=item['name'],features=left['features'],engine=engine,reference=ref_name,**metric(actual,reference,1e-4)))
    assert total_bytes==sum(math.prod(o['shape'])*8 for o in spec['outputs'])*len(jobs)
    for name in (['bridge'] if phase=='bridge' else ['bridge','remaining']):bindings[name+'.json']=pin(base/(name+'.json'))
    return dict(structural_passed=True,reference_passed=all_pass,phase=phase,manifest=manifest,jobs=len(jobs),arrays=len(jobs)*41,bytes=total_bytes,
        resources=summaries,reference_comparisons=comparisons,fp32_comparisons=fp32,files=bindings)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--phase',choices=['bridge','full'],required=True);args=parser.parse_args()
    base=Path(args.artifact).resolve();value=audit(base,args.phase);path=base/(args.phase+'-audit.json');write(path,value)
    if args.phase=='bridge' and value['reference_passed']:
        write(base/'bridge-gate.json',dict(passed=True,manifest=value['manifest'],files=value['files']|{'bridge-audit.json':pin(path)}))
    print(json.dumps(dict(structural_passed=True,reference_passed=value['reference_passed'],jobs=value['jobs'],arrays=value['arrays'],
        max_reference=max(v['max_scaled'] for v in value['reference_comparisons']),fp32_failed_arrays=sum(r['failed_values']>0 for r in value['fp32_comparisons']))))
