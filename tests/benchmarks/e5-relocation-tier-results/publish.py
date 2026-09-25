"""Publish the closed exact-product observation, without selecting scored calls."""
import bisect
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/e5-relocation-tier-diagnostic-amd-20260925'
TOOLS = ROOT/'tests/benchmarks/e5-relocation-tier-diagnostic-amd'
sys.path.insert(0, str(TOOLS))
from protocol import BLOCK, CALLS, KEYS, ROLES, pin, read

PREVIOUS = ROOT/'tests/benchmarks/e5-direct-tier-results/publish.py'
spec = importlib.util.spec_from_file_location('previous_tier_publisher', PREVIOUS)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
associations = previous.associations
timelines = previous.timelines
phase = previous.phase


def write_csv(name, rows):
    assert rows
    with (OUT/name).open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def namespace_loads(calls, events):
    starts = [c['begin_ms'] for c in calls]
    result = []
    for event in events:
        if event['provider'] != 'Microsoft-Windows-DotNETRuntime' or event['name'] != 'Method/LoadVerbose':
            continue
        p = event['payload']
        if not p['MethodNamespace'].startswith('Lokad.Onnx.'):
            continue
        preceding = bisect.bisect_right(starts, event['ms']) - 1
        inside = preceding if preceding >= 0 and event['ms'] <= calls[preceding]['end_ms'] else -1
        result.append(dict(event_index=event['index'], ms=event['ms'],
            since_first_call_ms=event['ms']-starts[0], preceding_call=preceding, inside_call=inside,
            thread=event['thread'], namespace=p['MethodNamespace'], method=p['MethodName'],
            signature=p['MethodSignature'], method_id=p['MethodID'], tier=p['OptimizationTier'],
            address=p['MethodStartAddress'], bytes=int(p['MethodSize'])))
    return result


def main():
    assert not (OUT/'observations-20260925.json').exists()
    proof = read(BASE/'closed.json')
    assert proof['passed'] and proof['diagnostic_only'] and not proof['release_admitted']
    assert pin(BASE/'closed.json')['sha256'] == '5b87937aa914ab9007928bfb3ba0f90c99eae3ff8f3d23611467cbb98ab8ed4b'
    for name, wanted in proof['files'].items():
        assert pin(BASE/name) == wanted, name
    analysis = read(BASE/'analysis.json')
    clocks, blocks, loads, compiles, pauses, summaries, reports = [], [], [], [], [], [], {}
    for role, report in analysis['reports'].items():
        assert role in ROLES
        value = read(BASE/f'collected/{role}-capture/output/result.json')
        observer = read(BASE/f'collected/{role}-capture/output/diagnostic.json')
        with gzip.open(BASE/f'collected/{role}-export/events/events.jsonl.gz', 'rt', encoding='utf8') as stream:
            events = [json.loads(line) for line in stream]
        assert len(events) == report['events']
        calls = report['calls']
        assoc = associations(calls, events)
        tiers = timelines(calls, events)
        namespace = namespace_loads(calls, events)
        assert len(namespace) == len(assoc['loads'])
        identity = dict(process=role, key=KEYS[role], product=ROLES[role])
        loads.extend(dict(**identity, **row) for row in namespace)
        compiles.extend(dict(**identity, **row) for row in assoc['compilations'])
        pauses.extend(dict(**identity, **row) for row in assoc['suspensions'])
        for clock, observed, call in zip(value['clocks'], observer['clocks'], calls, strict=True):
            assert call['phase'] == phase(call['index'])
            clocks.append(dict(**identity, phase=call['phase'], **clock,
                **{k: v for k, v in observed.items() if k != 'index'},
                begin_ms=call['begin_ms'], end_ms=call['end_ms']))
        extended = []
        for block in report['blocks']:
            group = calls[block['first']:block['last']+1]
            assert len(group) == BLOCK
            def overlap(rows):
                return sum(max(0, min(r['end_ms'], c['end_ms'])-max(r['start_ms'], c['begin_ms']))
                    for r in rows for c in group)
            row = dict(**identity, phase=phase(block['first']), **block,
                compilation_overlap_ms=overlap(assoc['compilations']),
                suspension_overlap_ms=overlap(assoc['suspensions']))
            for timeline in tiers:
                for boundary, at in [('start', group[0]['begin_ms']), ('end', group[-1]['end_ms'])]:
                    before = [t for t in timeline['timeline'] if t['ms'] <= at]
                    row[timeline['method']+'_'+boundary] = before[-1]['tier'] if before else 'not-yet-loaded'
            extended.append(row)
            blocks.append(row)
        optimized = [e for e in namespace if e['tier'] == 'OptimizedTier1']
        assert optimized
        last = max(optimized, key=lambda e: e['ms'])
        census = dict(**identity, core=value['core'],
            warmup_end_ms=calls[599]['end_ms']-calls[0]['begin_ms'],
            original_prefix_end_ms=calls[779]['end_ms']-calls[0]['begin_ms'],
            call_span_ms=calls[-1]['end_ms']-calls[0]['begin_ms'],
            namespace_loads=len(namespace), optimized_loads=len(optimized),
            last_optimized_load_ms=last['since_first_call_ms'], last_optimized_preceding_call=last['preceding_call'],
            loads_after_prefix=sum(e['ms'] > calls[779]['end_ms'] for e in namespace),
            loads_after_call_3000=sum(e['ms'] >= calls[3000]['begin_ms'] for e in namespace))
        summaries.append(census)
        reports[role] = dict(census=census, events=report['events'], clr_events=report['clr_events'],
            markers=report['markers'], blocks=extended, matrix_timelines=tiers,
            last_optimized_load=last, namespace_loads=namespace, **assoc)
    assert len(clocks) == 4*CALLS and len(blocks) == 4*CALLS//BLOCK
    assert sum(r['markers'] for r in reports.values()) == 8*CALLS
    write_csv('clocks-20260925.csv', clocks)
    write_csv('blocks-20260925.csv', blocks)
    write_csv('method-loads-20260925.csv', loads)
    write_csv('compilation-20260925.csv', compiles)
    write_csv('suspensions-20260925.csv', pauses)
    write_csv('runtime-phase-20260925.csv', summaries)
    inputs = [Path(__file__), PREVIOUS, previous.ASSOCIATIONS,
        ROOT/'artifacts/e5-relocation-tier-audit-import-20260925.json']
    result = dict(diagnostic_only=True, no_admission_score=True, release_admitted=False,
        closure=pin(BASE/'closed.json'), inputs={str(p.relative_to(ROOT)): pin(p) for p in inputs},
        resources=analysis['resources'], peak_rss=analysis['peak_rss'],
        compiled_review=analysis['compiled_review'], observer_review=analysis['observer_review'],
        products=analysis['products'], failed_release_cases=analysis['failed_release_cases'],
        reused_exporter=analysis['reused_exporter'], reports=reports)
    with (OUT/'observations-20260925.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    lines = ['# Exact relocation product: short-e5 runtime phases', '',
        'Four fresh processes ran release f95a13c5, relocation e07a4518, relocation,',
        'release under unchanged .NET 10.0.8 settings. Each completed 6,000 calls.',
        'The original 600 warmup and 180 measurement labels are retained; subsequent',
        'calls are a diagnostic extension. **Every clock is unscored.**', '',
        '| Process | Core | Warmup ends (s) | Original prefix ends (s) | Last namespace Tier1 load (s) | Loads after prefix | Loads after call 3,000 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        lines.append(f"| {s['process']} | {s['core'][:8]} | {s['warmup_end_ms']/1000:.6f} | "
            f"{s['original_prefix_end_ms']/1000:.6f} | {s['last_optimized_load_ms']/1000:.6f} | "
            f"{s['loads_after_prefix']} | {s['loads_after_call_3000']} |")
    lines += ['', 'The census includes every CLR Method/LoadVerbose event whose namespace starts',
        'with Lokad.Onnx. It is not a whole-CLR quiescence check. Exact matrix',
        'signatures and method identifiers are retained separately:', '',
        '| Process | Method | Optimized Tier1 seconds after first call |', '| --- | --- | ---: |']
    for role, report in reports.items():
        for timeline in report['matrix_timelines']:
            times = [t['since_first_call_ms']/1000 for t in timeline['timeline'] if t['tier'] == 'OptimizedTier1']
            rendered = ', '.join(f'{v:.6f}' for v in times) if times else 'none observed'
            lines.append(f"| {role} | {timeline['method']} | {rendered} |")
    lines += ['', 'All consecutive 30-call blocks are shown below. No block is omitted or scored.', '',
        '| Calls (zero-based) | Phase | Release a (ms) | Relocation b (ms) | Relocation c (ms) | Release d (ms) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for index in range(CALLS//BLOCK):
        group = [reports[r]['blocks'][index] for r in ROLES]
        assert len({(r['first'], r['last']) for r in group}) == 1
        lines.append(f"| {group[0]['first']}–{group[0]['last']} | {group[0]['phase']} | "
            + ' | '.join(f"{r['wall_ms']:.6f}" for r in group) + ' |')
    lines += ['', 'All 24,000 calls, 48,000 markers and original numerical arrays reconcile.',
        'Inputs remain immutable and retained outputs independently owned. Both reused',
        'compiled reviews pass; the observer and exporter were not rebuilt.',
        f"All {analysis['resources']:,} resource samples pass; peak owned RSS is {analysis['peak_rss']:,} bytes.",
        'Every recorded process is terminal and all streams report zero lost events.', '',
        'The first local audit command failed at import because its reused checks module',
        'was not yet on the Python search path. It wrote no audit result. Importing the',
        'frozen prepare module first supplied the existing path; the unchanged auditor',
        'then completed once. No capture, collection or product was repeated or changed.', '',
        'A load event establishes that a code version became available, not that every',
        'later invocation executed it. Instrumentation changes process history. These',
        'observations cannot establish the cause of the earlier uninstrumented failure',
        'or replace it. Compilation overlap is elapsed overlap, not compilation CPU cost.',
        'No release admission, selected-tail score or warmup-policy change is produced.', '',
        '[Every clock](clocks-20260925.csv), [all blocks and tier labels](blocks-20260925.csv),',
        '[namespace method loads with full identities](method-loads-20260925.csv),',
        '[compilation pairs](compilation-20260925.csv), [runtime suspensions](suspensions-20260925.csv),',
        '[phase census](runtime-phase-20260925.csv), [complete observations](observations-20260925.json).', '',
        'Raw evidence: artifacts/e5-relocation-tier-diagnostic-amd-20260925.',
        'Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    with (OUT/'report-20260925.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(closure=pin(BASE/'closed.json'), observations=pin(OUT/'observations-20260925.json'),
        census=summaries)))


if __name__ == '__main__':
    main()
