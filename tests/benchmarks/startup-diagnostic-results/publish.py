"""Publish every fixed block and runtime association without an admission score."""
import bisect,collections,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/benchmarks/startup-diagnostic-amd'))
from protocol import pin,read
BASE=ROOT/'artifacts/graph-startup-diagnostic-amd-20260923'

def csvfile(name,rows):
    assert rows
    with (OUT/name).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def associations(calls,events):
    starts=[c['begin_ms'] for c in calls];ends=[c['end_ms'] for c in calls]
    def overlap(start,end):
        i=max(0,bisect.bisect_left(ends,start));found=[]
        while i<len(calls) and starts[i]<end:
            delta=max(0,min(end,ends[i])-max(start,starts[i]))
            if delta:found.append((i,delta))
            i+=1
        return found
    def location(at):
        i=bisect.bisect_right(starts,at)-1
        return i if i>=0 and at<=ends[i] else -1
    queues=collections.defaultdict(list);loads=[];compiles=[];pending_pauses={};pauses=[];unmatched=[]
    for e in events:
        if e['provider']!='Microsoft-Windows-DotNETRuntime':continue
        p=e['payload']
        if e['name']=='Method/JittingStarted':queues[(e['thread'],p['MethodID'])].append(e)
        elif e['name']=='Method/LoadVerbose':
            key=(e['thread'],p['MethodID'])
            if queues[key]:
                start=queues[key].pop(0);touched=overlap(start['ms'],e['ms'])
                compiles.append(dict(method=p['MethodName'],namespace=p['MethodNamespace'],thread=e['thread'],tier=p['OptimizationTier'],
                    start_ms=start['ms'],end_ms=e['ms'],overlap_ms=sum(d for i,d in touched),calls=len(touched)))
            else:unmatched.append(dict(kind='load_without_start',event=e))
            if p['MethodNamespace'].startswith('Lokad.Onnx.'):
                loads.append(dict(ms=e['ms'],call=location(e['ms']),thread=e['thread'],method=p['MethodName'],namespace=p['MethodNamespace'],
                    tier=p['OptimizationTier'],bytes=int(p['MethodSize']),address=p['MethodStartAddress']))
        elif e['name']=='GC/SuspendEEStart':
            assert e['thread'] not in pending_pauses;pending_pauses[e['thread']]=e
        elif e['name']=='GC/RestartEEStop':
            start=pending_pauses.pop(e['thread'],None)
            if start is None:unmatched.append(dict(kind='restart_without_suspend',event=e));continue
            touched=overlap(start['ms'],e['ms'])
            pauses.append(dict(thread=e['thread'],start_ms=start['ms'],end_ms=e['ms'],reason=start['payload']['Reason'],
                overlap_ms=sum(d for i,d in touched),calls=len(touched)))
    unmatched += [dict(kind='start_without_load',event=e) for items in queues.values() for e in items]
    unmatched += [dict(kind='suspend_without_restart',event=e) for e in pending_pauses.values()]
    return dict(loads=loads,compilations=compiles,suspensions=pauses,unmatched=unmatched)

def main():
    proof=read(BASE/'closed.json');assert proof['passed'] and proof['diagnostic_only'] and not proof['root_product_changed']
    for n,v in proof['files'].items():assert pin(BASE/n)==v,n
    a=read(BASE/'analysis.json');clocks=[];blocks=[];loads=[];compiles=[];pauses=[];reports={}
    for role,report in a['reports'].items():
        value=read(BASE/f'collected/{role}-capture/output/result.json')
        events=[json.loads(line) for line in (BASE/f'collected/{role}-export/events/events.jsonl').read_text().splitlines()]
        assoc=associations(report['calls'],events)
        for name,destination in [('loads',loads),('compilations',compiles),('suspensions',pauses)]:
            destination.extend(dict(process=role,**r) for r in assoc[name])
        clocks.extend(dict(process=role,**clock,begin_ms=c['begin_ms'],end_ms=c['end_ms']) for clock,c in zip(value['clocks'],report['calls'],strict=True))
        blocks.extend(dict(process=role,**r) for r in report['blocks'])
        reports[role]=dict(events=report['events'],clr_events=report['clr_events'],markers=report['markers'],
            blocks=report['blocks'],**assoc)
    csvfile('clocks-20260923.csv',clocks);csvfile('blocks-20260923.csv',blocks)
    csvfile('method-loads-20260923.csv',loads);csvfile('compilation-20260923.csv',compiles)
    if pauses:csvfile('suspensions-20260923.csv',pauses)
    result=dict(diagnostic_only=True,no_admission_score=True,closure=pin(BASE/'closed.json'),
        resources=a['resources'],peak_rss=a['peak_rss'],reports=reports)
    (OUT/'observations-20260923.json').open('x',encoding='utf8').write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Unchanged GPT-2 runtime diagnostic','',
        'Two fresh processes use the unchanged selected release (Core `521bae17`),',
        '1,200 calls each, normal runtime settings, CPU 2 on AMD EPYC 9V74.',
        'The graph execution and all original numerical/ownership checks remain',
        'unchanged. CPU 0 collects CLR events and request markers. No sample profiler.',
        'These instrumented clocks cannot replace the inconclusive M43 score.','',
        '| Calls, zero-based | Process A wall ms | Process A CPU ms | Process B wall ms | Process B CPU ms |',
        '| --- | ---: | ---: | ---: | ---: |']
    for x,y in zip(reports['a']['blocks'],reports['b']['blocks'],strict=True):
        assert (x['first'],x['last'])==(y['first'],y['last'])
        lines.append(f"| {x['first']}–{x['last']} | {x['wall_ms']:.6f} | {x['cpu_ms']:.6f} | {y['wall_ms']:.6f} | {y['cpu_ms']:.6f} |")
    lines+=['','Every call appears in a fixed consecutive block; none is dropped or',
        'selected as a favorable steady-state window. CPU counters are coarse',
        'process totals and include work by runtime threads; event capture adds',
        'overhead. Compilation/suspension overlap is an association, not proof of',
        'the cause of the original uninstrumented process difference.','',
        f"Both processes preserve all original output arrays, native error limits, inputs and held results. All 4,800 request markers reconcile; neither event stream reports loss. All {a['resources']:,} resource observations pass; peak owned RSS {a['peak_rss']:,} bytes. All owners are terminal.",'',
        '[Every clock](clocks-20260923.csv), [every block](blocks-20260923.csv),',
        '[product method loads](method-loads-20260923.csv),',
        '[paired compilation intervals](compilation-20260923.csv),',
        '[complete observations and unmatched boundary events](observations-20260923.json).','',
        'No source change, release admission or Microsoft ORT comparison follows',
        'from this diagnostic. Raw traces and every exported event remain in the',
        'ignored local artifact directory.','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    (OUT/'report-20260923.md').open('x',encoding='utf8').write('\n'.join(lines)+'\n')
    print(json.dumps({k:dict(events=r['events'],compilations=len(r['compilations']),suspensions=len(r['suspensions']),
        unmatched=len(r['unmatched']),last_product_load=r['loads'][-1]) for k,r in reports.items()}))

if __name__=='__main__':main()
