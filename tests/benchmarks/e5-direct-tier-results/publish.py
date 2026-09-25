"""Publish every call/block and exact dispatcher tier timeline from closed evidence."""
import bisect
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-direct-tier-diagnostic-v2-amd'))
from protocol import CALLS, BLOCK, ROLES, KEYS, pin, read
ASSOCIATIONS=ROOT/'tests/benchmarks/e5-runtime-diagnostic-results/associations.py'
sys.path.insert(0,str(ASSOCIATIONS.parent))
from associations import associations
BASE=ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'
NAMESPACE='Lokad.Onnx.Tensor`1[System.Single]'
SIGNATURES={
    'RunFloatMatMulKernel':'void  (int32,int32,int32,float32*,float32*,float32*,value class Lokad.Onnx.TensorExecutionOptions)',
    'RunBatchedFloatMatMul':'void  (class Lokad.Onnx.Tensor`1<float32>,class Lokad.Onnx.Tensor`1<float32>,class Lokad.Onnx.Tensor`1<float32>,value class Lokad.Onnx.TensorExecutionOptions)',
}


def phase(index):
    return 'warmup' if index<600 else 'original-measured-label' if index<780 else 'diagnostic-extension'


def csvfile(name,rows):
    assert rows
    with (OUT/name).open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def timelines(calls,events):
    starts=[c['begin_ms'] for c in calls];result=[]
    for method,signature in SIGNATURES.items():
        selected=[e for e in events if e['provider']=='Microsoft-Windows-DotNETRuntime'
                  and e['name']=='Method/LoadVerbose' and e['payload']['MethodNamespace']==NAMESPACE
                  and e['payload']['MethodName']==method]
        assert selected and {e['payload']['MethodSignature'] for e in selected}=={signature}
        identities={e['payload']['MethodID'] for e in selected};assert len(identities)==1
        rows=[]
        for event in selected:
            p=event['payload'];preceding=bisect.bisect_right(starts,event['ms'])-1
            inside=preceding if preceding>=0 and event['ms']<=calls[preceding]['end_ms'] else -1
            rows.append(dict(ms=event['ms'],since_first_call_ms=event['ms']-starts[0],
                inside_call=inside,preceding_call=preceding,tier=p['OptimizationTier'],
                address=p['MethodStartAddress'],bytes=int(p['MethodSize']),raw_event=event))
        result.append(dict(method=method,namespace=NAMESPACE,signature=signature,method_id=next(iter(identities)),
            final_tier1_observed=any(r['tier']=='OptimizedTier1' for r in rows),timeline=rows))
    return result


def main():
    assert not (OUT/'observations-20260925.json').exists()
    proof=read(BASE/'closed.json');assert proof['passed'] and proof['diagnostic_only'] and not proof['root_product_changed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');clocks=[];blocks=[];loads=[];compiles=[];pauses=[];reports={}
    for role,report in analysis['reports'].items():
        assert role in ROLES
        value=read(BASE/f'collected/{role}-capture/output/result.json')
        observation=read(BASE/f'collected/{role}-capture/output/diagnostic.json')
        with gzip.open(BASE/f'collected/{role}-export/events/events.jsonl.gz','rt',encoding='utf8') as stream:
            events=[json.loads(line) for line in stream]
        assert len(events)==report['events']
        assoc=associations(report['calls'],events);tiers=timelines(report['calls'],events)
        for name,destination in [('loads',loads),('compilations',compiles),('suspensions',pauses)]:
            destination.extend(dict(process=role,key=KEYS[role],product=ROLES[role],**r) for r in assoc[name])
        for clock,observed,call in zip(value['clocks'],observation['clocks'],report['calls'],strict=True):
            assert call['phase']==phase(call['index'])
            clocks.append(dict(process=role,key=KEYS[role],product=ROLES[role],phase=call['phase'],**clock,
                **{k:v for k,v in observed.items() if k!='index'},begin_ms=call['begin_ms'],end_ms=call['end_ms']))
        extended=[]
        for block in report['blocks']:
            calls=report['calls'][block['first']:block['last']+1];assert len(calls)==BLOCK
            def overlap(rows):
                return sum(max(0,min(row['end_ms'],c['end_ms'])-max(row['start_ms'],c['begin_ms'])) for row in rows for c in calls)
            row=dict(process=role,key=KEYS[role],product=ROLES[role],phase=phase(block['first']),**block,
                compilation_overlap_ms=overlap(assoc['compilations']),suspension_overlap_ms=overlap(assoc['suspensions']))
            for timeline in tiers:
                for boundary,at in [('start',calls[0]['begin_ms']),('end',calls[-1]['end_ms'])]:
                    before=[t for t in timeline['timeline'] if t['ms']<=at]
                    row[timeline['method']+'_'+boundary]=before[-1]['tier'] if before else 'not-yet-loaded'
            extended.append(row);blocks.append(row)
        reports[role]=dict(key=KEYS[role],product=ROLES[role],events=report['events'],clr_events=report['clr_events'],
            markers=report['markers'],call_span_ms=report['calls'][-1]['end_ms']-report['calls'][0]['begin_ms'],
            blocks=extended,matrix_timelines=tiers,**assoc)
    assert len(clocks)==4*CALLS and len(blocks)==4*CALLS//BLOCK and sum(r['markers'] for r in reports.values())==8*CALLS
    csvfile('clocks-20260925.csv',clocks);csvfile('blocks-20260925.csv',blocks)
    if loads:csvfile('method-loads-20260925.csv',loads)
    if compiles:csvfile('compilation-20260925.csv',compiles)
    if pauses:csvfile('suspensions-20260925.csv',pauses)
    result=dict(diagnostic_only=True,no_admission_score=True,closure=pin(BASE/'closed.json'),
        inputs={str(p.relative_to(ROOT)):pin(p) for p in [Path(__file__),ASSOCIATIONS]},
        resources=analysis['resources'],peak_rss=analysis['peak_rss'],compiled_review=analysis['compiled_review'],
        observer_review=analysis['observer_review'],products=analysis['products'],
        failed_release_cases=analysis['failed_release_cases'],reused_exporter=analysis['reused_exporter'],
        release_admitted=False,reports=reports)
    with (OUT/'observations-20260925.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    lines=['# Exact-candidate short-e5 tier observation','',
        'Four fresh CPU-2 processes run release, candidate, candidate, release with',
        '6,000 calls each under unchanged .NET 10.0.8 settings. The first 780 calls',
        'preserve 600 warmup and 180 original measurement labels; later calls are',
        'an unscored diagnostic extension. Every clock remains diagnostic.','',
        '| Process | Product | Method | Final Tier1 observed | Final Tier1 seconds after first call |',
        '|---|---|---|---|---:|']
    for role,report in reports.items():
        for timeline in report['matrix_timelines']:
            times=[t['since_first_call_ms']/1000 for t in timeline['timeline'] if t['tier']=='OptimizedTier1']
            text=', '.join(f'{v:.6f}' for v in times) if times else '—'
            lines.append(f"| {role} | {ROLES[role]} | {timeline['method']} | {timeline['final_tier1_observed']} | {text} |")
    lines+=['','All consecutive 30-call blocks follow. No block is omitted or scored.','',
        '| Calls, zero-based | Phase | Release A, ms | Candidate B, ms | Candidate C, ms | Release D, ms |',
        '|---|---|---:|---:|---:|---:|']
    for index in range(CALLS//BLOCK):
        group=[reports[r]['blocks'][index] for r in ROLES]
        assert len({(r['first'],r['last']) for r in group})==1
        lines.append(f"| {group[0]['first']}–{group[0]['last']} | {group[0]['phase']} | "+' | '.join(f"{r['wall_ms']:.6f}" for r in group)+' |')
    lines+=['','All 24,000 calls and 48,000 markers reconcile with the raw event streams.',
        'The original numerical, immutable-input and independently held-output',
        'checks pass. Both compiled reviews pass: reversing four observation hooks',
        'and the loop constant recovers the original benchmark; reversing exactly',
        'two constants recovers the complete previous observer. No product or',
        'runtime option changed. The proven lossless exporter was reused.','',
        f"All {analysis['resources']:,} resource samples pass; peak owned RSS is {analysis['peak_rss']:,} bytes.",
        'Every recorded owner and worker is terminal; no event stream reports loss.','',
        'Tier loads are identified by exact method signature and runtime identifier.',
        'A block label describes the most recently loaded method version; events do',
        'not prove which code version executed on every call. Compilation overlap is',
        'elapsed overlap, not compilation CPU cost. Instrumentation changes runtime',
        'history; these results cannot retroactively explain the older uninstrumented',
        'regression or replace the failed release comparison.','',
        '[Every clock](clocks-20260925.csv), [all blocks and tier labels](blocks-20260925.csv),',
        '[product method loads](method-loads-20260925.csv), [compilation pairs](compilation-20260925.csv),',
        '[suspensions](suspensions-20260925.csv), [exact timelines and complete observations](observations-20260925.json).','',
        'Raw traces and every decoded event remain under',
        '`artifacts/e5-direct-tier-diagnostic-v2-amd-20260925`. No release admission or',
        'warmup-policy change follows from this diagnostic.','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    with (OUT/'report-20260925.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(closure=pin(BASE/'closed.json'),observations=pin(OUT/'observations-20260925.json'),
        processes={k:dict(events=r['events'],span_ms=r['call_span_ms'],unmatched=len(r['unmatched']),
            tiers=[dict(method=t['method'],loads=[(x['tier'],x['since_first_call_ms'],x['inside_call']) for x in t['timeline']]) for t in r['matrix_timelines']]) for k,r in reports.items()})))


if __name__=='__main__':main()
