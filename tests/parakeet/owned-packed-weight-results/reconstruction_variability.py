"""Localize failed reconstruction controls in retained clocks; never trim or rerun."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-owned-packed-weight-reconstruction-cost-amd-20260925'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    assert pin(BASE / 'closed.json')['sha256'] == '1750ff9e337a06840f1001b335ac211909bc721c09abc33f8aabc389e46c8f38'
    closure = read(BASE / 'closed.json')
    assert closure['passed'] and not closure['usable_for_attribution']
    assert closure['analysis'] == pin(BASE / 'analysis.json')
    analysis = read(BASE / 'analysis.json')
    failed = [r for r in analysis['summary']['controls'] if not r['passed']]
    assert len(failed) == 3 and {r['role'] for r in failed} == {'candidate'}
    assert {(r['name'], r['metric']) for r in failed} == {
        ('260-123286-0000', 'feed_forward'), ('260-123286-0000', 'copy_y'), ('complete-corpus', 'copy_y')}
    rows = []
    focused = []
    sources = {}
    for job in ['candidate-512-a', 'candidate-512-b']:
        name = 'capture-collected/probe/' + job + '/result.json'
        path = BASE / name
        assert pin(path) == closure['files'][name]
        sources[name] = pin(path)
        result = read(path)
        assert result['passed'] and len(result['records']) == 80
        for record in result['records']:
            calls = record['calls']
            assert len(calls) == 96
            maximum = max(calls, key=lambda c: c['copy_y_ticks'])
            row = dict(job=job, request=record['request_index'], name=record['name'], frames=record['frames'],
                       pass_index=record['pass'], phase=record['phase'],
                       encoder_seconds=(record['encoder_end']-record['encoder_start'])/record['encoder_frequency'],
                       copy_seconds=sum(c['copy_y_ticks'] for c in calls)/10000000,
                       math_seconds=sum(c['math_ticks'] for c in calls)/10000000,
                       reconstructions=sum(c['copy_y_stages'] for c in calls),
                       largest_copy_seconds=maximum['copy_y_ticks']/10000000,
                       largest_copy_node=maximum['node'] if maximum['copy_y_ticks'] else '',
                       accompanying_math_seconds=maximum['math_ticks']/10000000 if maximum['copy_y_ticks'] else 0)
            rows.append(row)
            if row['name'] == '260-123286-0000':
                focused.append(dict(**row, calls=calls))
    assert len(rows) == 160 and len(focused) == 8
    assert sum(r['reconstructions'] for r in rows) == 4872
    assert all(r['frames'] == 89 and r['reconstructions'] == 87 for r in focused)
    paths = [OUT / ('reconstruction-variability-20260925' + suffix) for suffix in ['.json', '.csv', '.md']]
    assert not any(path.exists() for path in paths)
    text = '''# Reconstruction timing: localizing the failed controls

The completed diagnostic passes output, ownership, traffic and resource checks,
but only 89/92 timing controls. Quantitative attribution remains rejected. This
read-only review uses every retained candidate request and changes no score.
All three failed controls concern candidate CopyY or complete feed-forward time
for `260-123286-0000`, an 89-frame clip, and the resulting corpus CopyY total.

| Process | Pass | Phase | Encoder seconds | CopyY seconds | MatMul Math seconds | Largest single CopyY seconds |
|---|---:|---|---:|---:|---:|---:|
'''
    for r in focused:
        text += f"| {r['job']} | {r['pass_index']} | {r['phase']} | {r['encoder_seconds']:.6f} | {r['copy_seconds']:.6f} | {r['math_seconds']:.6f} | {r['largest_copy_seconds']:.6f} |\n"
    spike, = [r for r in focused if r['job'] == 'candidate-512-a' and r['pass_index'] == 1]
    text += f'''
In the first measured pass of candidate A, reconstruction stages sum to
{spike['copy_seconds']:.6f} s. The largest stage is
`{spike['largest_copy_node']}`: {spike['largest_copy_seconds']:.6f} s, alongside
{spike['accompanying_math_seconds']:.6f} s of Math in that same call.
The request contains exactly the same 87 full 16 MiB reconstructions as its
other passes. The JSON preserves all 96 calls for each of the eight rows above;
the CSV preserves all 160 candidate requests, including warmup and unaffected clips.

The clocks place the delay inside the existing reconstruction/allocation stage.
They do not identify garbage collection, allocation stalls, scheduling or page
faults as the cause. No runtime events were captured, and this boundary includes
preparation before Math. Do not label the delay as GC without observing it.
Do not remove the slow pass, lengthen warmup, rerun unchanged timing or relax
the failed thresholds.

The next bounded test follows the verified source difference: prove that the
existing final-row arithmetic can consume packed weights directly, removing
the known full-matrix reconstruction. Failed timing controls prevent a stable
cost or projected saving claim; a fresh application comparison must establish
the benefit. The runtime cause of these spikes remains unproved.

If event correlation becomes necessary, retain the full corpus order because
prior requests determine heap state. Existing stage durations have no absolute
boundaries; new hooks require an explicitly identified diagnostic build and an
observer-effect check. GC attribution is not a prerequisite for proving identical
arithmetic that avoids the verified redundant copies. No kernel or cache sweep
is selected.

[Closed diagnostic](reconstruction-20260925.md),
[all candidate requests](reconstruction-variability-20260925.csv),
[focused calls and source identities](reconstruction-variability-20260925.json).
The failed application and e5 release controls remain unchanged.
'''
    with paths[0].open('x', encoding='utf8') as stream:
        json.dump(dict(passed=True, read_only=True, closure=pin(BASE/'closed.json'), analysis=pin(BASE/'analysis.json'),
                       sources=sources, failed_controls=failed, focused=focused, all_requests=rows,
                       quantitative_attribution=False, runtime_cause_proved=False, clocks_trimmed=False,
                       new_inference=False, new_variant_selected=False, reviewer=pin(Path(__file__))),
                  stream, indent=2, allow_nan=False)
    with paths[1].open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paths[2].open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)
    print(json.dumps(dict(passed=True, requests=len(rows), focused_requests=len(focused),
                          failed_controls=len(failed), runtime_cause_proved=False,
                          largest_copy_seconds=spike['largest_copy_seconds'])))


if __name__ == '__main__':
    main()
