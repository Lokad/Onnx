"""Publish complete diagnostic clocks and runtime associations, without scoring."""
import bisect
import csv
import json
from pathlib import Path
import statistics
import sys
from associations import associations

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tests/parakeet/pad-runtime-diagnostic-amd'))
from protocol import pin, read

BASE = ROOT / 'artifacts/parakeet-pad-runtime-diagnostic-amd-20260923'


def csvfile(name, rows):
    assert rows
    with (OUT / name).open('x', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def per_call_overlap(calls, intervals):
    """Sum interval overlaps, preserving potential overlap between compiler threads."""
    starts = [c['begin_ms'] for c in calls]; ends = [c['end_ms'] for c in calls]
    totals = [0.0] * len(calls)
    for interval in intervals:
        start, end = interval['start_ms'], interval['end_ms']
        index = bisect.bisect_left(ends, start)
        while index < len(calls) and starts[index] < end:
            totals[index] += max(0, min(end, ends[index]) - max(start, starts[index]))
            index += 1
    return totals


def main():
    proof = read(BASE / 'closed.json')
    assert proof['passed'] and proof['diagnostic_only'] and not proof['root_product_changed']
    for name, wanted in proof['files'].items(): assert pin(BASE / name) == wanted, name
    analysis = read(BASE / 'analysis.json')
    clocks = []; blocks = []; loads = []; compiles = []; pauses = []; reports = {}
    for role, report in analysis['reports'].items():
        value = read(BASE / f'collected/{role}-capture/result.json')
        events = [json.loads(line) for line in (BASE / f'collected/{role}-export/events/events.jsonl').read_text().splitlines()]
        assoc = associations(report['calls'], events)
        for row in assoc['loads']:
            index = max(row['call'], row['previous_call'])
            call = report['calls'][index] if index >= 0 else None
            row['case'] = None if call is None else call['name']
            row['iteration'] = None if call is None else call['iteration']
        compilation = per_call_overlap(report['calls'], assoc['compilations'])
        suspension = per_call_overlap(report['calls'], assoc['suspensions'])
        for name, destination in [('loads', loads), ('compilations', compiles), ('suspensions', pauses)]:
            destination.extend(dict(process=role, **row) for row in assoc[name])
        summaries = []
        for case in value['rows']:
            index = case['index']; start = index * 780
            case_calls = report['calls'][start:start + 780]
            for call, clock in zip(case_calls, case['clocks'], strict=True):
                clocks.append(dict(process=role, case=index, name=case['name'], **clock,
                    begin_ms=call['begin_ms'], end_ms=call['end_ms'],
                    compilation_overlap_ms=compilation[call['index']], suspension_overlap_ms=suspension[call['index']]))
            selected = case_calls[600:]
            summaries.append(dict(case=index, name=case['name'], diagnostic_mean_ms=statistics.mean(c['wall_ms'] for c in selected),
                first_ms=case_calls[0]['begin_ms'], last_ms=case_calls[-1]['end_ms'],
                measured_wall_ms=sum(c['wall_ms'] for c in selected),
                measured_suspension_ms=sum(suspension[start + 600:start + 780]),
                measured_compilation_ms=sum(compilation[start + 600:start + 780]),
                median_thread_allocation_bytes=statistics.median(c['allocated_bytes'] for c in selected),
                measured_collection_calls=sum(any(c[g] for g in ['gc0', 'gc1', 'gc2']) for c in selected)))
        for block in report['blocks']:
            start = block['case'] * 780 + block['first']; end = start + 60
            blocks.append(dict(process=role, **block,
                compilation_overlap_ms=sum(compilation[start:end]), suspension_overlap_ms=sum(suspension[start:end])))
        reports[role] = dict(events=report['events'], clr_events=report['clr_events'], markers=report['markers'],
            first_ms=report['calls'][0]['begin_ms'], last_ms=report['calls'][-1]['end_ms'], cases=summaries, **assoc)
    for name, rows in [('clocks', clocks), ('blocks', blocks), ('method-loads', loads), ('compilation', compiles), ('suspensions', pauses)]:
        csvfile(name + '-20260923.csv', rows)
    result = dict(diagnostic_only=True, no_admission_score=True, closure=pin(BASE / 'closed.json'),
        generators={p.name: pin(p) for p in [OUT / 'publish.py', OUT / 'associations.py']},
        resources=analysis['resources'], peak_rss=analysis['peak_rss'], reports=reports)
    (OUT / 'observations-20260923.json').open('x', encoding='utf8').write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Parakeet padding runtime diagnostic', '',
        'The selected process loads a fully optimized `PadCore` at 4.845 seconds,',
        'during attention-167 call 31. The candidate first reaches `PadCore` at',
        '1.812 seconds, at cropping call 0, and loads only optimized loop versions',
        'before completion. Its public Pad and dispatcher also have no fully',
        'optimized whole-method load in this trace. Both products use ordinary',
        '.NET 10.0.8 settings; their original PadCore bodies and flags are identical.', '',
        'The shorter execution reaches fallback work in a different compilation',
        'state. This supports investigating optimized compilation of the fallback',
        'on first use. It does not establish the cause of every original failed',
        'control or prove that a new compiler annotation would qualify for release.', '',
        '| Fallback case | Selected diagnostic ms | Candidate diagnostic ms | Selected suspension ms, 180 calls | Candidate suspension ms, 180 calls |',
        '| --- | ---: | ---: | ---: | ---: |']
    for a, b in zip(reports['current']['cases'][9:], reports['candidate']['cases'][9:], strict=True):
        lines.append(f"| {a['name']} | {a['diagnostic_mean_ms']:.6f} | {b['diagnostic_mean_ms']:.6f} | {a['measured_suspension_ms']:.6f} | {b['measured_suspension_ms']:.6f} |")
    lines += ['',
        'The candidate median request-thread allocations are 88 bytes higher for',
        'each fallback case. Counters include marker instrumentation; they do not',
        'isolate product allocations. Reflection has no measured collection calls',
        'or suspension overlap in either process. Garbage-collection pauses alone',
        'therefore cannot explain its diagnostic difference. Primary fast-copy',
        'blocks still vary; all blocks and collection/compilation associations',
        'are retained instead of selecting a favorable subset.', '',
        'One selected and one candidate process execute the unchanged twelve-case',
        'workload, 780 calls per case. CPU 2 runs requests on AMD EPYC 9V74; CPU 0',
        'collects compilation/GC events and markers. The diagnostic consumer adds',
        'instrumentation around the same public call, oracle and ownership checks.',
        'The 600/180 labels identify original boundaries, not a new score. Primary',
        'shapes are synthetic and use recorded frame counts, not captured tensors.', '',
        f"All 18,720 calls and 37,440 markers reconcile with zero event loss. All {sum(r['events'] for r in reports.values()):,} events remain, including events outside requests. Every output bit, input and independent held output passes. All {analysis['resources']} resource observations pass, peak owned RSS {analysis['peak_rss']:,} bytes; all owners are terminal.", '',
        'Compilation and suspension overlap use request-marker boundaries, which',
        'include marker overhead. Compilation intervals on different threads can',
        'overlap; their summed durations are not exclusive CPU time or attribution.',
        'Method-load events show availability, not instruction-by-instruction',
        'execution. Instrumentation changes timing and runtime history.', '',
        '[Every clock](clocks-20260923.csv), [all 312 blocks](blocks-20260923.csv),',
        '[product method loads](method-loads-20260923.csv),',
        '[compilation intervals](compilation-20260923.csv),',
        '[suspensions](suspensions-20260923.csv),',
        '[all observations and unmatched boundary events](observations-20260923.json).', '',
        'The [original M47 screen](../pad-dispatch-results/screen-20260923.md)',
        'remains rejected. No application comparison, ORT parity claim, product',
        'integration or benchmark-table change follows this diagnostic.', '',
        'Closure: `' + pin(BASE / 'closed.json')['sha256'] + '`.']
    (OUT / 'report-20260923.md').open('x', encoding='utf8').write('\n'.join(lines) + '\n')
    print(json.dumps({k: dict(events=v['events'], compilations=len(v['compilations']), suspensions=len(v['suspensions']),
        unmatched=len(v['unmatched']), first_ms=v['first_ms'], last_ms=v['last_ms'], fallback=v['cases'][9:]) for k, v in reports.items()}))


if __name__ == '__main__': main()
