"""Publish every e5 call and runtime association, with no release-admission score."""
import csv
import gzip
import json
from pathlib import Path
import sys


ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-repeatability-diagnostic-amd'))
ASSOCIATIONS=ROOT/'tests/benchmarks/e5-runtime-diagnostic-results/associations.py'
sys.path.insert(0,str(ASSOCIATIONS.parent))
from associations import associations
from protocol import ROLES,KEYS,pin,read
BASE=ROOT/'artifacts/e5-repeatability-diagnostic-amd-20260925'


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
        with gzip.open(BASE/f'collected/{role}-export/events/events.jsonl.gz','rt',encoding='utf8') as stream:
            events=[json.loads(line) for line in stream]
        assoc=associations(report['calls'],events)
        for name,destination in [('loads',loads),('compilations',compiles),('suspensions',pauses)]:
            destination.extend(dict(process=role,key=KEYS[role],product=ROLES[role],**r) for r in assoc[name])
        for clock,observed,call in zip(value['clocks'],observation['clocks'],report['calls'],strict=True):
            clocks.append(dict(process=role,key=KEYS[role],product=ROLES[role],**clock,**{k:v for k,v in observed.items() if k!='index'},begin_ms=call['begin_ms'],end_ms=call['end_ms']))
        extended=[]
        for block in report['blocks']:
            calls=report['calls'][block['first']:block['last']+1]
            def overlap(rows):
                return sum(max(0,min(row['end_ms'],c['end_ms'])-max(row['start_ms'],c['begin_ms'])) for row in rows for c in calls)
            row=dict(process=role,key=KEYS[role],product=ROLES[role],warmup=block['first']<600,**block,
                compilation_overlap_ms=overlap(assoc['compilations']),suspension_overlap_ms=overlap(assoc['suspensions']))
            extended.append(row);blocks.append(row)
        reports[role]=dict(key=KEYS[role],product=ROLES[role],events=report['events'],clr_events=report['clr_events'],markers=report['markers'],
            blocks=extended,**assoc)
    assert len(clocks)==6240 and len(blocks)==104 and sum(r['markers'] for r in reports.values())==12480
    csvfile('clocks-20260925.csv',clocks);csvfile('blocks-20260925.csv',blocks)
    if loads:csvfile('method-loads-20260925.csv',loads)
    if compiles:csvfile('compilation-20260925.csv',compiles)
    if pauses:csvfile('suspensions-20260925.csv',pauses)
    result=dict(diagnostic_only=True,no_admission_score=True,closure=pin(BASE/'closed.json'),
        inputs={str(p.relative_to(ROOT)):pin(p) for p in [Path(__file__),ASSOCIATIONS]},
        resources=analysis['resources'],peak_rss=analysis['peak_rss'],compiled_review=analysis['compiled_review'],
        products=analysis['products'],failed_release_controls=analysis['failed_release_controls'],
        roundtrip=analysis['roundtrip'],release_admitted=False,reports=reports)
    with (OUT/'observations-20260925.json').open('x',encoding='utf8') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Diagnose the two failed e5 repeatability controls','',
        'Eight fresh processes observe the exact qualified and isolated M73 products.',
        'Each input runs selected, candidate, candidate, selected on CPU 2 with',
        'normal .NET 10.0.8 settings. Every original benchmark statement remains;',
        'four existing observation hooks record call boundaries and process',
        'CPU/allocation/collection counters outside the timer. CPU 0 collects events.',
        'These diagnostic clocks do not replace the failed release comparison.','']
    for key,roles in [('e5-8tok','abcd'),('e5-512tok','efgh')]:
        lines += ['## '+key,'',
            '| Calls, zero-based | Selected A, ms | Candidate B, ms | Candidate C, ms | Selected D, ms |',
            '|---|---:|---:|---:|---:|']
        for index in range(13):
            group=[reports[role]['blocks'][index] for role in roles]
            assert len({(r['first'],r['last']) for r in group})==1 and all(reports[role]['key']==key for role in roles)
            lines.append(f"| {group[0]['first']}–{group[0]['last']} | "+' | '.join(f"{r['wall_ms']:.6f}" for r in group)+' |')
        lines.append('')
    lines += ['All 6,240 calls, 600 warmup labels per process and 12,480 boundary markers',
        'are retained. Original numerical, immutable-input and independently held',
        'output checks pass; hashes match the uninstrumented comparison. No event',
        'stream reports loss. Compiled review preserves all original method flags',
        'and instructions, branches, locals and exception regions after removing',
        'the four observation hooks. Products are reused without rebuilding.','',
        'Every decoded event remains in lossless gzip storage. Before capture,',
        'the exporter reproduced all bytes of a complete retained event export.',
        f"All {analysis['resources']:,} resource observations pass; peak owned RSS is {analysis['peak_rss']:,} bytes.",
        'Every owner and worker is terminal.','',
        'Compilation overlap is elapsed overlap with a compilation start/load pair;',
        'it is not compilation CPU cost or a causal allocation of latency. Process',
        'counters include runtime threads. Instrumentation changes runtime history',
        'and cannot retroactively attribute the original failed timing difference.',
        'Every unmatched event and every call remain retained.','',
        '[Every clock](clocks-20260925.csv), [all fixed blocks and overlaps](blocks-20260925.csv),',
        '[method loads](method-loads-20260925.csv), [compilation pairs](compilation-20260925.csv),',
        '[suspensions](suspensions-20260925.csv), [complete observations](observations-20260925.json).','',
        'Raw traces and all decoded events remain under',
        '`artifacts/e5-repeatability-diagnostic-amd-20260925`.',
        'The original failed comparison is',
        '[retained separately](../../parakeet/slice-dense-conversion-results/e5-repeatability-20260925.md).',
        'No release admission or new warmup policy follows from publication alone.','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    with (OUT/'report-20260925.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:dict(events=r['events'],compilations=len(r['compilations']),suspensions=len(r['suspensions']),
        unmatched=len(r['unmatched']),last_product_load=r['loads'][-1] if r['loads'] else None) for k,r in reports.items()}))


if __name__=='__main__':main()
