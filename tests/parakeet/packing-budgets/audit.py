"""Audit complete trajectories, original native failures and bounded resources."""
import importlib.util
import math
from common import *


def main():
    prepared=read(BASE/'prepared.json');assert prepared['passed'];verify(prepared['files'])
    for p in (QUALIFIED/'closed.json',NATIVE_BASELINE/'closed.json'):verify(read(p)['files'])
    state=read(BASE/'processes.json');assert state['complete'] and state['code']==0;terminal(state['supervisor'])
    assert [r['name'] for r in state['runs']]==[f'{mib}-{mode}' for mib in (512,2032) for mode in ('trace','public','native')]
    for row in state['runs']:
        assert row['complete'] and row['application_passed'] and row['samples']>0
        for pid,birth in row['members'].items():terminal(dict(pid=int(pid),birth=birth))
        samples=[json.loads(line) for line in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples'] and max(s['rss'] for s in samples)==row['peak_rss']
        assert row['preflight']['available']>=14*1024**3
        assert all(s['seconds']<1200 and s['rss']<12*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3
            and len(s['members'])<=1 and all(p['affinity']==[2] for p in s['members']) and s['output_bytes']<=1024**3 for s in samples)
    spec=read(MANIFEST);original=QUALIFIED/'trace-output';old=read(original/'result.json')
    native_source=ROOT/'tests/parakeet/transcribe/audit.py'
    loader=importlib.util.spec_from_file_location('original_native_audit',native_source);native_module=importlib.util.module_from_spec(loader);loader.loader.exec_module(native_module)
    baseline_native=native_module.audit(REFERENCE,NATIVE_BASELINE/'baseline.json')
    baseline_result=read(NATIVE_BASELINE/'baseline.json')
    known={'english-16k/step-26/outputs','english-frame-limit/step-26/outputs','english-repeat/step-26/outputs'}
    def failures(report):return {'/'.join(r[k] for k in ('case','label','output')):r for r in report['failures']}
    assert set(failures(baseline_native))==known and not baseline_native['numeric_gate_passed']
    summaries=[]
    for mib in (512,2032):
        variant=prepared['variants'][str(mib)];candidate=BASE/f'{mib}-trace/output';public_path=BASE/f'{mib}-public/output'
        trace=read(candidate/'result.json');public=read(public_path/'result.json')
        for result in (trace,public):
            assert result['passed'] and result['error'] is None and result['inputs_and_held_outputs_unchanged']
            assert result['flags']=={} and result['affinity']==4 and result['processor_count']==1 and result['runtime']=='.NET 10.0.12'
            assert result['manifest_sha256']==pin(MANIFEST)['sha256']
            assert result['core_sha256']==variant['core']['sha256'] and result['data_sha256']==variant['data']['sha256'] and result['runner_sha256']==variant['profile']['sha256']
        assert trace['mode']=='trace' and public['mode']=='public' and public['call_files']==[] and trace['applications']==[]
        assert trace['call_files']==old['call_files']==[f'{i:04}.json' for i in range(1240)]
        assert read(candidate/'graphs.json')==read(original/'graphs.json')==read(public_path/'graphs.json')
        assert len(trace['traced'])==len(public['applications'])==len(spec['cases'])==20
        for traced,application,c in zip(trace['traced'],public['applications'],spec['cases'],strict=True):
            assert traced['name']==application['name']==c['name'] and traced['pass']==0
            assert traced['result']==application['result']==c['expected']
        counters=dict(graph_calls=0,input_arrays=0,output_arrays=0,values=0);seen=set()
        for name in trace['call_files']:
            before=read(original/name);after=read(candidate/name)
            for key in ('name','graph','step','pass','nodes'):assert after[key]==before[key],(name,key)
            for group,counter in [('inputs','input_arrays'),('outputs','output_arrays')]:
                assert before[group].keys()==after[group].keys()
                for key,value in after[group].items():
                    reference=before[group][key]
                    for field in ('dtype','shape','values','sha256'):assert value[field]==reference[field],(mib,name,group,key,field)
                    assert value['dtype'] in ('<f4','<i4','<i8') and value['values']==math.prod(value['shape'])
                    path=(candidate/value['file']).resolve();assert path.is_relative_to((candidate/'arrays').resolve()) and value['file'] not in seen;seen.add(value['file'])
                    actual=pin(path);assert actual['sha256']==value['sha256'] and actual['bytes']==value['values']*(8 if value['dtype']=='<i8' else 4)
                    assert actual==pin(original/reference['file'])
                    counters[counter]+=1;counters['values']+=value['values']
            counters['graph_calls']+=1
        assert counters==dict(graph_calls=1240,input_arrays=6080,output_arrays=4880,values=28831376)
        packing=read(candidate/'packing.json');assert packing==read(public_path/'packing.json')
        for graph_name,graph in packing.items():
            assert graph['maximum_packed_bytes']==dict(frontend=0,encoder=mib*1024**2,decoder=64*1024**2)[graph_name]
            assert sum(w['bytes'] for w in graph['weights'])==graph['retained_packed_bytes']<=graph['maximum_packed_bytes']
            assert all(w['bytes']==math.prod(w['shape'])*4 for w in graph['weights'])
        assert packing['encoder']['retained_packed_bytes']==variant['retained_bytes'] and len(packing['encoder']['weights'])==variant['weights']
        native_path=BASE/f'{mib}-native/result.json';native_result=read(native_path)
        assert native_result['core_sha256']==variant['core']['sha256'] and native_result['data_sha256']==variant['data']['sha256']
        assert native_result['runner_sha256']==variant['native']['sha256'] and native_result['runtime']=='.NET 10.0.12' and not native_result['settings']
        native_report=native_module.audit(REFERENCE,native_path);save(BASE/f'{mib}-native-audit.json',native_report)
        assert native_report['audit_consistent'] and native_report['application_passed'] and not native_report['numeric_gate_passed']
        assert failures(native_report)==failures(baseline_native) and native_report['maximum']==baseline_native['maximum']
        native_arrays=0;native_values=0
        for a,b in zip(native_result['rows'],baseline_result['rows'],strict=True):
            assert a['name']==b['name'] and a['actual']==b['actual']
            for x,y in zip(a['comparisons'],b['comparisons'],strict=True):
                for key in ('label','output','shape','dtype','file'):assert x[key]==y[key]
                left=Path(str(native_path)+'.tensors')/x['file'];right=NATIVE_BASELINE/'baseline.json.tensors'/y['file']
                assert pin(left)==pin(right),(mib,a['name'],x['label'],x['output'])
                native_arrays+=1;native_values+=math.prod(x['shape'])
        assert native_arrays==784 and native_values==3090494
        native_run=next(r for r in state['runs'] if r['name']==f'{mib}-native');assert native_run['code']==1 and native_run['native_numeric_passed'] is False
        summaries.append(dict(budget_mib=mib,trace=counters,public_requests=20,native_arrays=native_arrays,native_values=native_values,
            regression_passed=True,native_numeric_passed=False,native_failures=native_report['failures'],maximum_native_error=native_report['maximum'],
            retained_bytes=variant['retained_bytes'],retained_weights=variant['weights'],resources=[r for r in state['runs'] if r['name'].startswith(str(mib)+'-')]))
    files=dict(prepared['files'])
    for p in BASE.rglob('*'):
        if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts):files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.iterdir():
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    files[native_source.relative_to(ROOT).as_posix()]=pin(native_source)
    assert not (BASE/'closed.json').exists()
    save(BASE/'closed.json',dict(passed=True,regression_passed=True,native_numeric_passed=False,files=files,variants=summaries,
        scope='Higher-budget local complete inference and memory qualification; original native numerical failures unchanged; no speed or AMD claim'))
    print(json.dumps(dict(passed=True,variants=[{k:v for k,v in s.items() if k!='resources'} for s in summaries],closed=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
