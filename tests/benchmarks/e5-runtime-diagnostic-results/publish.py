"""Publish every e5 call and runtime association, with no release-admission score."""
import csv
import json
from pathlib import Path
import sys
from associations import associations

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-runtime-diagnostic-amd'))
from protocol import ROLES,pin,read
BASE=ROOT/'artifacts/e5-runtime-diagnostic-amd-20260924'


def csvfile(name,rows):
    assert rows
    with (OUT/name).open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    proof=read(BASE/'closed.json');assert proof['passed'] and proof['diagnostic_only'] and not proof['root_product_changed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');clocks=[];blocks=[];loads=[];compiles=[];pauses=[];reports={}
    for role,report in analysis['reports'].items():
        assert role in ROLES
        value=read(BASE/f'collected/{role}-capture/output/result.json')
        observation=read(BASE/f'collected/{role}-capture/output/diagnostic.json')
        events=[json.loads(line) for line in (BASE/f'collected/{role}-export/events/events.jsonl').read_text().splitlines()]
        assoc=associations(report['calls'],events)
        for name,destination in [('loads',loads),('compilations',compiles),('suspensions',pauses)]:
            destination.extend(dict(process=role,product=ROLES[role],**r) for r in assoc[name])
        for clock,observed,call in zip(value['clocks'],observation['clocks'],report['calls'],strict=True):
            clocks.append(dict(process=role,product=ROLES[role],**clock,**{k:v for k,v in observed.items() if k!='index'},begin_ms=call['begin_ms'],end_ms=call['end_ms']))
        extended=[]
        for block in report['blocks']:
            calls=report['calls'][block['first']:block['last']+1]
            def overlap(rows):
                return sum(max(0,min(row['end_ms'],c['end_ms'])-max(row['start_ms'],c['begin_ms'])) for row in rows for c in calls)
            row=dict(process=role,product=ROLES[role],warmup=block['first']<600,**block,
                compilation_overlap_ms=overlap(assoc['compilations']),suspension_overlap_ms=overlap(assoc['suspensions']))
            extended.append(row);blocks.append(row)
        reports[role]=dict(product=ROLES[role],events=report['events'],clr_events=report['clr_events'],markers=report['markers'],
            blocks=extended,**assoc)
    assert len(clocks)==3120 and len(blocks)==52 and sum(r['markers'] for r in reports.values())==6240
    csvfile('clocks-20260924.csv',clocks);csvfile('blocks-20260924.csv',blocks)
    if loads:csvfile('method-loads-20260924.csv',loads)
    if compiles:csvfile('compilation-20260924.csv',compiles)
    if pauses:csvfile('suspensions-20260924.csv',pauses)
    result=dict(diagnostic_only=True,no_admission_score=True,closure=pin(BASE/'closed.json'),
        inputs={str(p.relative_to(ROOT)):pin(p) for p in [Path(__file__),OUT/'associations.py']},
        resources=analysis['resources'],peak_rss=analysis['peak_rss'],compiled_review=analysis['compiled_review'],reports=reports)
    with (OUT/'observations-20260924.json').open('x',encoding='utf8') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Focused e5 runtime diagnostic','',
        'Four fresh processes run selected, candidate, candidate, selected with',
        'unchanged M66 product binaries, normal .NET 10.0.8 settings and CPU 2.',
        'Every original warmed-consumer statement remains; four observation hooks',
        'add event markers and CPU/allocation/collection counters outside its timer.',
        'CPU 0 collects and exports runtime events. Instrumentation changes runtime',
        'history: these clocks do not qualify a release or replace the failed score.','',
        '| Calls, zero-based | Selected A, ms | Candidate B, ms | Candidate C, ms | Selected D, ms |',
        '|---|---:|---:|---:|---:|']
    for index in range(13):
        group=[reports[role]['blocks'][index] for role in ROLES]
        assert len({(r['first'],r['last']) for r in group})==1
        lines.append(f"| {group[0]['first']}–{group[0]['last']} | "+' | '.join(f"{r['wall_ms']:.6f}" for r in group)+' |')
    lines+=['','All 3,120 calls, 600 warmup labels per process and 6,240 markers are retained.',
        'Every original numerical, input and held-output check passes; output hashes',
        'match the uninstrumented comparison. No event stream reports loss. The',
        'compiled review preserves all original method flags and every original',
        'instruction, branch, local and exception region after removing the four hooks.','',
        f"All {analysis['resources']:,} resource observations pass; peak owned RSS is {analysis['peak_rss']:,} bytes. Every owner is terminal.",'',
        'Compilation overlap is elapsed overlap with a compilation start/load pair;',
        'it is not measured compilation CPU cost or a causal allocation of latency.',
        'Counters cover the process, including runtime threads. Full unmatched',
        'boundary events remain in the observations. No call is trimmed.','',
        '[Every clock](clocks-20260924.csv), [all fixed blocks and overlaps](blocks-20260924.csv),',
        '[method loads](method-loads-20260924.csv), [compilation pairs](compilation-20260924.csv),',
        '[suspensions](suspensions-20260924.csv), [complete observations](observations-20260924.json).','',
        'Raw nettrace files and every decoded event remain under',
        '`artifacts/e5-runtime-diagnostic-amd-20260924`. The original failed comparison',
        'is [retained separately](../../parakeet/validated-composition-results/graphs-20260924.md).','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    with (OUT/'report-20260924.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:dict(events=r['events'],compilations=len(r['compilations']),suspensions=len(r['suspensions']),
        unmatched=len(r['unmatched']),last_product_load=r['loads'][-1] if r['loads'] else None) for k,r in reports.items()}))


if __name__=='__main__':main()
