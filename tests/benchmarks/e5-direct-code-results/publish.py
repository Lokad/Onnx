"""Publish complete code listings, event associations and all diagnostic clocks."""
import csv
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-direct-code-diagnostic-amd'))
from protocol import CALLS,BLOCK,ROLES,DIAGNOSTIC_FLAGS,read,pin
ASSOCIATIONS=ROOT/'tests/benchmarks/e5-runtime-diagnostic-results/associations.py'
sys.path.insert(0,str(ASSOCIATIONS.parent))
from associations import associations
BASE=ROOT/'artifacts/e5-direct-code-diagnostic-amd-20260925'


def phase(index):return 'warmup' if index<600 else 'original-measured-label' if index<780 else 'diagnostic-extension'


def csvfile(name,rows):
    assert rows
    with (OUT/name).open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    assert not (OUT/'observations-20260925.json').exists()
    proof=read(BASE/'closed.json');assert proof['passed'] and proof['diagnostic_only'] and not proof['release_admitted']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');assert analysis['diagnostic_flags']==DIAGNOSTIC_FLAGS
    clocks=[];blocks=[];loads=[];compilations=[];pauses=[];reports={};code=[]
    native=OUT/'listings-20260925';native.mkdir()
    for role,report in analysis['reports'].items():
        assert role in ROLES
        value=read(BASE/f'collected/{role}-capture/output/result.json');diagnostic=read(BASE/f'collected/{role}-capture/output/diagnostic.json')
        with gzip.open(BASE/f'collected/{role}-export/events/events.jsonl.gz','rt',encoding='utf8') as f:events=[json.loads(line) for line in f]
        assert len(events)==report['events'];assoc=associations(report['calls'],events)
        for name,destination in [('loads',loads),('compilations',compilations),('suspensions',pauses)]:
            destination.extend(dict(process=role,product=ROLES[role],**row) for row in assoc[name])
        for clock,observed,call in zip(value['clocks'],diagnostic['clocks'],report['calls'],strict=True):
            assert call['phase']==phase(call['index'])
            clocks.append(dict(process=role,product=ROLES[role],phase=call['phase'],**clock,
                **{k:v for k,v in observed.items() if k!='index'},begin_ms=call['begin_ms'],end_ms=call['end_ms']))
        extended=[]
        for block in report['blocks']:
            calls=report['calls'][block['first']:block['last']+1];assert len(calls)==BLOCK
            def overlap(rows):return sum(max(0,min(r['end_ms'],c['end_ms'])-max(r['start_ms'],c['begin_ms'])) for r in rows for c in calls)
            row=dict(process=role,product=ROLES[role],phase=phase(block['first']),**block,
                compilation_overlap_ms=overlap(assoc['compilations']),suspension_overlap_ms=overlap(assoc['suspensions']))
            blocks.append(row);extended.append(row)
        versions=[]
        for row in analysis['codegen'][role]['rows']:
            path=native/f"{role}-{row['index']:02d}-{row['method']}.asm"
            with path.open('x',encoding='utf8',newline='\n') as f:f.write(row['listing']+'\n')
            version=dict(process=role,product=ROLES[role],**{k:v for k,v in row.items() if k!='listing'},
                listing_path=path.relative_to(OUT).as_posix(),listing_file=pin(path),
                since_first_call_seconds=(row['event']['ms']-report['calls'][0]['begin_ms'])/1000)
            versions.append(version);code.append(version)
        reports[role]=dict(product=ROLES[role],pid=value['pid'],events=report['events'],markers=report['markers'],
            blocks=extended,code_versions=versions,**assoc)
    assert len(clocks)==4*CALLS and len(blocks)==4*CALLS//BLOCK
    csvfile('clocks-20260925.csv',clocks);csvfile('blocks-20260925.csv',blocks)
    csvfile('method-loads-20260925.csv',loads);csvfile('compilation-20260925.csv',compilations);csvfile('suspensions-20260925.csv',pauses)
    result=dict(diagnostic_only=True,no_score=True,release_admitted=False,closure=pin(BASE/'closed.json'),
        parent_closure=analysis['parent_closure'],products=analysis['products'],consumer=analysis['consumer'],
        diagnostic_flags=DIAGNOSTIC_FLAGS,compiled_review=analysis['compiled_review'],resources=analysis['resources'],peak_rss=analysis['peak_rss'],
        inputs={str(p.relative_to(ROOT)):pin(p) for p in [Path(__file__),ASSOCIATIONS]},reports=reports)
    with (OUT/'observations-20260925.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    lines=['# Exact e5 dispatcher listings and runtime events','',
        'The exact qualified release and direct-depthwise candidate run in four',
        'fresh processes: release, candidate, candidate, release. Each executes',
        '6,000 e5-8tok calls with the original 600 warmup and 180 measurement labels',
        'preserved in the prefix. All clocks, including the extension, are diagnostic.','',
        'Only two logging variables are set: DOTNET_JitDisasm selects the two matrix',
        'dispatchers and DOTNET_JitDisasmWithCodeBytes prints instruction bytes.',
        'Optimization, tiering, PGO and SIMD settings remain unchanged. The consumer',
        'changes only to accept those fixed logging values: 130 of its 131 original',
        'methods are unchanged, and Main recovers after removing its old/new flag',
        'predicates. Numerical, timing, input and held-output checks remain intact.','',
        'Each complete listing matches its own process\'s runtime method identity,',
        'tier, order and native-code size. Logging can perturb runtime history;',
        'these instructions cannot be attached to an earlier process. The index',
        'retains byte columns as emitted, native-size metadata and complete listings.','',
        '| Process | Method | Tier | Native bytes | Printed instruction bytes | Load seconds | Listing |',
        '|---|---|---|---:|---:|---:|---|']
    for row in code:
        lines.append(f"| {row['process']} | {row['method']} | {row['tier']} | {row['native_bytes']} | {row['instruction_bytes']} | {row['since_first_call_seconds']:.6f} | [assembly]({row['listing_path']}) |")
    lines+=['',f"All {analysis['resources']:,} resource samples pass; peak owned RSS is {analysis['peak_rss']:,} bytes.",
        'All 24,000 calls and 48,000 markers reconcile, with zero reported event loss.',
        'Every original numerical and ownership check passes. Every owner is terminal.','',
        '[Every clock](clocks-20260925.csv), [all 800 fixed blocks](blocks-20260925.csv),',
        '[product code loads](method-loads-20260925.csv), [compilation pairs](compilation-20260925.csv),',
        '[suspensions](suspensions-20260925.csv), [complete code/event associations](observations-20260925.json).','',
        'Raw traces, full stdout and every decoded event remain under',
        '`artifacts/e5-direct-code-diagnostic-amd-20260925`. Existing failed release',
        'verdicts, warmup policy and BENCHMARK.md are unchanged. This is evidence',
        'for instruction inspection, not a new application score.','',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    with (OUT/'report-20260925.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(closure=pin(BASE/'closed.json'),observations=pin(OUT/'observations-20260925.json'),
        versions=len(code),per_process={r:dict(unmatched=len(v['unmatched']),code=[dict(method=x['method'],tier=x['tier'],native_bytes=x['native_bytes'],instruction_bytes=x['instruction_bytes']) for x in v['code_versions']]) for r,v in reports.items()})))


if __name__=='__main__':main()
