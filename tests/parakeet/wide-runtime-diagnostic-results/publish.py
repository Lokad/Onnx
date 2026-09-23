"""Publish all diagnostic intervals without changing the rejected M52 score."""
import bisect
import csv
import json
from pathlib import Path
import statistics
import sys
from associations import associations

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tests/parakeet/wide-runtime-diagnostic-amd'))
from protocol import pin, read
BASE = ROOT / 'artifacts/parakeet-wide-runtime-diagnostic-amd-20260923'


def csvfile(name, rows):
    assert rows
    with (OUT / name).open('x', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def per_call_overlap(calls, intervals):
    """Sum overlaps; simultaneous compiler threads can count the same time twice."""
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
            row['case'] = None if call is None else call['case']
            row['iteration'] = None if call is None else call['iteration']
        compilation = per_call_overlap(report['calls'], assoc['compilations'])
        suspension = per_call_overlap(report['calls'], assoc['suspensions'])
        for name, destination in [('loads', loads), ('compilations', compiles), ('suspensions', pauses)]:
            destination.extend(dict(process=role, **row) for row in assoc[name])
        summaries = []
        for case in value['rows']:
            index = case['index']; start = index * 120
            case_calls = report['calls'][start:start + 120]
            for call, clock in zip(case_calls, case['clocks'], strict=True):
                clocks.append(dict(process=role, case=index, name=case['name'], **clock,
                    begin_ms=call['begin_ms'], end_ms=call['end_ms'],
                    compilation_overlap_ms=compilation[call['index']], suspension_overlap_ms=suspension[call['index']]))
            selected = case_calls[60:]
            summaries.append(dict(case=index, name=case['name'], m=case['m'], reduction=case['reduction'], columns=case['columns'],
                diagnostic_mean_ms=statistics.mean(c['wall_ms'] for c in selected),
                first_ms=case_calls[0]['begin_ms'], last_ms=case_calls[-1]['end_ms'],
                measured_wall_ms=sum(c['wall_ms'] for c in selected),
                measured_suspension_ms=sum(suspension[start + 60:start + 120]),
                measured_compilation_ms=sum(compilation[start + 60:start + 120]),
                median_process_allocation_bytes=statistics.median(c['total_allocated_bytes'] for c in selected),
                measured_collection_calls=sum(any(c[g] for g in ['gc0', 'gc1', 'gc2']) for c in selected)))
        for block in report['blocks']:
            start = block['case'] * 120 + block['first']; end = start + 20
            blocks.append(dict(process=role, **block,
                compilation_overlap_ms=sum(compilation[start:end]), suspension_overlap_ms=sum(suspension[start:end])))
        reports[role] = dict(events=report['events'], clr_events=report['clr_events'], markers=report['markers'],
            first_ms=report['calls'][0]['begin_ms'], last_ms=report['calls'][-1]['end_ms'], cases=summaries, **assoc)
    for name, rows in [('clocks', clocks), ('blocks', blocks), ('method-loads', loads), ('compilation', compiles), ('suspensions', pauses)]:
        csvfile(name + '-20260923.csv', rows)
    result = dict(diagnostic_only=True, no_admission_score=True, closure=pin(BASE / 'closed.json'),
        generators={p.name: pin(p) for p in [OUT / 'publish.py', OUT / 'associations.py']},
        resources=analysis['resources'], peak_rss=analysis['peak_rss'], reports=reports,
        tables={p.name: pin(p) for p in OUT.glob('*-20260923.csv')})
    (OUT / 'observations-20260923.json').open('x', encoding='utf8').write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: dict(events=v['events'], compilations=len(v['compilations']), suspensions=len(v['suspensions']),
        unmatched=len(v['unmatched']), first_ms=v['first_ms'], last_ms=v['last_ms'], failed_geometry=v['cases'][5]) for k, v in reports.items()}))


if __name__ == '__main__': main()
