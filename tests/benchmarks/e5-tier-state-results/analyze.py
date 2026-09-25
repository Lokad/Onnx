"""Read retained events to identify matrix-dispatch tiers, without new inference."""
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/e5-repeatability-diagnostic-amd-20260925'
PUBLISHED=OUT.parent/'e5-repeatability-diagnostic-results/observations-20260925.json'
NAMESPACE='Lokad.Onnx.Tensor`1[System.Single]'
METHODS=['RunFloatMatMulKernel','RunBatchedFloatMatMul']


def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    assert not (OUT/'tiers-20260925.json').exists()
    closed=read(BASE/'closed.json');analysis=read(BASE/'analysis.json');published=read(PUBLISHED)
    assert pin(BASE/'closed.json')['sha256']=='fe8ec805d18864a3a8d8b5a052543a3eef466535adc2769efc8c6c6747338a7b'
    assert closed['passed'] and closed['diagnostic_only'] and closed['analysis']==pin(BASE/'analysis.json')
    assert published['closure']==pin(BASE/'closed.json') and published['products']==analysis['products']
    inputs={str(p.relative_to(ROOT)):pin(p) for p in [Path(__file__),BASE/'closed.json',BASE/'analysis.json',PUBLISHED]}
    rows=[]
    for role in 'abcdefgh':
        report=analysis['reports'][role];observation=published['reports'][role]
        calls=report['calls'];assert len(calls)==780 and report['markers']==1560 and not observation['unmatched']
        path=BASE/f'collected/{role}-export/events/events.jsonl.gz'
        assert pin(path)==closed['files'][path.relative_to(BASE).as_posix()]
        inputs[path.relative_to(ROOT).as_posix()]=pin(path)
        selected=[];count=0
        with gzip.open(path,'rt',encoding='utf8') as stream:
            for line in stream:
                e=json.loads(line);count+=1;p=e['payload']
                if e['provider']=='Microsoft-Windows-DotNETRuntime' and e['name']=='Method/LoadVerbose' and p['MethodNamespace']==NAMESPACE and p['MethodName'] in METHODS:
                    selected.append(e)
        assert count==report['events']
        starts=[c['begin_ms'] for c in calls]
        for name in METHODS:
            events=[e for e in selected if e['payload']['MethodName']==name]
            assert len({e['payload']['MethodID'] for e in events})==1
            assert len({e['payload']['MethodSignature'] for e in events})==1
            expected=[r for r in observation['loads'] if r['namespace']==NAMESPACE and r['method']==name]
            assert len(expected)==len(events)>0
            for e,r in zip(events,expected,strict=True):
                p=e['payload']
                assert e['ms']==r['ms'] and p['OptimizationTier']==r['tier'] and p['MethodStartAddress']==r['address']
            timeline=[]
            for e,r in zip(events,expected,strict=True):
                preceding=bisect.bisect_right(starts,e['ms'])-1
                timeline.append(dict(ms=e['ms'],since_first_call_ms=e['ms']-starts[0],
                    inside_call=r['call'],preceding_call=preceding,tier=r['tier'],address=r['address'],bytes=r['bytes'],raw_event=e))
            rows.append(dict(process=role,input=observation['key'],product=observation['product'],
                method=name,namespace=NAMESPACE,method_id=events[0]['payload']['MethodID'],
                call_count=780,call_span_ms=calls[-1]['end_ms']-starts[0],timeline=timeline,
                all_product_tiers=dict(collections.Counter(r['tier'] for r in observation['loads'])),
                final_tier1_observed=any(r['tier']=='OptimizedTier1' for r in timeline)))
    assert len(rows)==16
    assert all(not r['final_tier1_observed'] for r in rows if r['input']=='e5-8tok')
    assert all(r['final_tier1_observed'] for r in rows if r['input']=='e5-512tok')
    result=dict(passed=True,diagnostic_only=True,new_inference=False,measurement_policy_changed=False,
        applies_to_products=published['products'],does_not_establish_m78_cause=True,inputs=inputs,rows=rows)
    with (OUT/'tiers-20260925.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(result=pin(OUT/'tiers-20260925.json'),rows=len(rows),
        short_final_tier1=sum(r['final_tier1_observed'] for r in rows if r['input']=='e5-8tok'),
        long_final_tier1=sum(r['final_tier1_observed'] for r in rows if r['input']=='e5-512tok'))))


if __name__=='__main__':main()
