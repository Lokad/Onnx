"""Describe all retained clocks for the two failed e5 controls; never rescore them."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-slice-dense-conversion-graphs-amd-20260925'
OUT = Path(__file__).resolve().parent
CASES = ['e5-8tok', 'e5-512tok']
ORDER = ['current-a', 'candidate-a', 'ort-a', 'ort-b', 'candidate-b', 'current-b']


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def mean(clocks):
    return sum(Fraction(row['ticks'], row['frequency']) for row in clocks) / len(clocks)


def main():
    closure = pin(BASE / 'closed.json')
    assert closure['sha256'] == 'e04e3a6a7434afd4af6bda1901c934004fc263db82a52f453f4321ca3f7a9fbd'
    proof = read(BASE / 'closed.json')
    assert proof['passed'] and not proof['admitted'] and not proof['all_controls_passed']
    assert pin(BASE / 'analysis.json') == proof['files']['analysis.json']
    analysis = read(BASE / 'analysis.json')
    assert analysis['passed'] and not analysis['root_product_changed']
    failed = [(case['key'], control['role']) for case in analysis['performance']
              for control in case['controls'] if not control['passed']]
    assert failed == [(case, 'candidate') for case in CASES]
    rows = []
    processes = []
    inputs = {}
    for case in CASES:
        score, = [row for row in analysis['performance'] if row['key'] == case]
        assert not score['qualified'] and score['regression_passed']
        for role in ORDER:
            name = f'collected/timing-{case}-{role}/output/result.json'
            path = BASE / name
            inputs[name] = pin(path)
            assert inputs[name] == proof['files'][name]
            result = read(path)
            clocks = result['clocks']
            assert result['passed'] and result['key'] == case and len(clocks) == 780
            assert [row['index'] for row in clocks] == list(range(780))
            assert all(row['warmup'] == (row['index'] < 600) for row in clocks)
            assert all(row['ticks'] > 0 and row['frequency'] > 0 for row in clocks)
            assert float(mean(clocks[600:])) == score['means'][role]
            blocks = []
            for start in range(0, 780, 30):
                group = clocks[start:start+30]
                exact = mean(group)
                row = dict(case=case, process=role, start=start, end=start+29,
                           warmup=start < 600, count=len(group), mean_seconds=float(exact),
                           numerator=exact.numerator, denominator=exact.denominator)
                blocks.append(row)
                rows.append(row)
            assert sum(row['count'] for row in blocks) == 780
            measured = blocks[20:]
            assert sum(Fraction(row['numerator'], row['denominator']) for row in measured)/6 == mean(clocks[600:])
            processes.append(dict(case=case, process=role, source=inputs[name],
                                  core=result.get('core'), consumer=result.get('consumer'),
                                  full_measured_mean=score['means'][role],
                                  measured_blocks=[row['mean_seconds'] for row in measured]))
    assert len(processes) == 12 and len(rows) == 312 and sum(row['count'] for row in rows) == 9360
    paths = [OUT / ('e5-repeatability-20260925' + suffix) for suffix in ['.json', '.csv', '.md']]
    assert not any(path.exists() for path in paths)
    lines = ['# Retained e5 repeatability failures: clock review', '',
             'The two failed controls remain failed. This read-only analysis uses all',
             '9,360 retained calls from the twelve processes for e5-8tok and e5-512tok.',
             'Fixed blocks contain 30 consecutive calls, including every warmup; no',
             'clock is trimmed and no replacement performance score is produced.', '',
             'Candidate process means differ by 11.980% at eight tokens and 13.229%',
             'at 512 tokens, exceeding the original 10% limit. All eight regression',
             'comparisons passed, but that does not override the repeatability failure.', '',
             'At eight tokens, the selected processes and candidate B have late changes',
             'within the measured window. At 512 tokens, candidate B is consistently',
             'faster than candidate A across all six measured blocks and continues to',
             'decline. The existing clocks cannot identify JIT compilation, collection',
             'or scheduling as the cause. They have call indices and elapsed ticks,',
             'but no absolute call boundaries to join to runtime events.', '',
             'A previous [30-token runtime diagnostic](../../benchmarks/e5-runtime-diagnostic-results/report-20260924.md)',
             'observed late compilation, but used different products and inputs.',
             'It supports investigating runtime state; it does not explain these',
             'failures or justify another warmup increase without observation.', '',
             'If release work proceeds, the next bounded diagnostic should preserve',
             'both products, both failing inputs, all 780 calls and their order, and',
             'observe runtime compilation, collection pauses and process CPU at call',
             'boundaries outside the existing timer. Include both products twice.',
             'Treat observer timings as diagnostic only and preserve the failed score.',
             'Do not overlap the active Parakeet application comparison.', '',
             'All times below are milliseconds. Columns are the six consecutive',
             '30-call blocks of the original 180-call measured window.', '',
             '| Case | Process | Full mean | 600–629 | 630–659 | 660–689 | 690–719 | 720–749 | 750–779 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in processes:
        numbers = [row['full_measured_mean'], *row['measured_blocks']]
        lines.append('| '+row['case']+' | '+row['process']+' | '+' | '.join(f'{x*1000:.3f}' for x in numbers)+' |')
    lines += ['', '[All fixed blocks, including warmup](e5-repeatability-20260925.csv),',
              '[source pins, original controls and process summaries](e5-repeatability-20260925.json).', '',
              'Original closure: `'+closure['sha256']+'`. No VM workload or product edit was made.']
    with paths[0].open('x', encoding='utf8') as stream:
        json.dump(dict(closure=closure, analysis=pin(BASE / 'analysis.json'), inputs=inputs,
                       failed_controls=failed, original_performance=analysis['performance'],
                       processes=processes, clocks=9360, release_admitted=False,
                       new_performance_score=False, reviewer=pin(Path(__file__))), stream, indent=2, allow_nan=False)
    with paths[1].open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paths[2].open('x', encoding='utf8', newline='\n') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True, clocks=9360, processes=12, blocks=312,
                          failed_controls=failed, new_score=False)))


if __name__ == '__main__':
    main()
