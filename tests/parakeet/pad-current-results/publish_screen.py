"""Publish the unchanged score and all fixed prefix/suffix blocks, including failures."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pad-current-screen-amd-20260926'
sys.path.insert(0, str(ROOT / 'tests/parakeet/pad-current-screen'))
from protocol import pin, read
from prefix import validate
from score import ORDER, score


def main():
    proof = read(BASE / 'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(BASE / name) == wanted, name
    analysis = read(BASE / 'analysis.json')
    reports = {name: read(BASE / 'collected' / name / 'result.json') for name in ORDER}
    for key, value in score(reports).items(): assert analysis[key] == value, key
    assert proof['admitted'] == analysis['admitted']
    blocks = []; setups = []; prefix_calls = 0; suffix_calls = 0
    for name, result in reports.items():
        prefix = read(BASE / 'collected' / name / 'priming.json')
        assert validate(prefix, result) == analysis['prefixes'][name]
        prefix_calls += prefix['calls']; suffix_calls += result['calls']
        groups = [('prefix', group['round'], group['rows']) for group in prefix['passes']]
        groups.append(('suffix', None, result['rows']))
        for phase, round_index, rows in groups:
            for row in rows:
                setups.append(dict(process=name, phase=phase, round=round_index,
                    case=row['name'], ticks=row['setupTicks'], frequency=result['frequency']))
                for first in range(0, 780, 60):
                    mean = Fraction(sum(c['ticks'] for c in row['clocks'][first:first + 60]), 60 * result['frequency'])
                    blocks.append(dict(process=name, phase=phase, round=round_index, case=row['name'],
                        first=first, last=first + 59, measured=phase == 'suffix' and first >= 600,
                        numerator=mean.numerator, denominator=mean.denominator, seconds=float(mean)))
    assert suffix_calls == 37440 and sum(b['measured'] for b in blocks) == 144
    assert len(blocks) * 60 == prefix_calls + suffix_calls
    for filename, rows in [('screen-blocks-20260926.csv', blocks), ('screen-setups-20260926.csv', setups)]:
        with (OUT / filename).open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    summary = dict(closure=pin(BASE / 'closed.json'), generator=pin(Path(__file__)),
        prefix_calls=prefix_calls, total_calls=prefix_calls + suffix_calls, fixed_blocks=len(blocks), **analysis)
    with (OUT / 'screen-observations-20260926.json').open('x', encoding='utf8') as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    verdict = 'passes all fixed gates' if analysis['admitted'] else 'fails its fixed gates; no model trial is admitted'
    lines = ['# Current-root padding: complete public-call screen', '',
        f'**The screen {verdict}.**', '',
        'Selected Core f3992f40 / Data a8e0b583; candidate Core a74acb17 / Data be954dc4.',
        'Both are the actual qualified build products. No root code or release',
        'benchmark changes follow this component result alone.', '',
        '| Synthetic case | Selected ms | Candidate ms | Candidate / selected | Regression gate |',
        '|---|---:|---:|---:|---|']
    for row in analysis['rows']:
        lines.append(f"| {row['name']} | {row['current']['value'] * 1000:.6f} | {row['candidate']['value'] * 1000:.6f} | {row['ratio']['value']:.6f} | {'Pass' if row['passed'] else 'Fail'} |")
    lines += ['', f"Primary eight-case reduction: {(1 - analysis['eligible']['ratio']['value']) * 100:.6f}%.",
        f"Repeatability: {sum(r['passed'] for r in analysis['controls'])}/26 controls pass (limit 1.10).",
        f"All-case regression: {sum(r['passed'] for r in analysis['rows'])}/12 pass (limit 1.05).", '']
    for check in analysis['controls']:
        if not check['passed']:
            lines.append(f"Failed repeatability: {check['role']}, case {check['case']}, ratio {check['ratio']['value']:.6f}.")
    for gate in analysis['gates']:
        lines.append(f"Gate {gate['name']}: {'pass' if gate['passed'] else 'fail'}.")
    lines += ['', 'Each fresh process runs the same full-census prefix, stopping at the first',
        'round ending at least ten seconds after the first round ended. Then the',
        'original twelve-case 600/180 suffix runs unchanged. All prefix calls remain',
        'warmup; every original suffix clock contributes at its prescribed position.', '',
        '| Process | Prefix rounds | Prefix calls | Seconds after first census |', '|---|---:|---:|---:|']
    for name, row in analysis['prefixes'].items():
        lines.append(f"| {name} | {row['rounds']} | {row['calls']} | {row['seconds_after_first']:.6f} |")
    lines += ['', f'{prefix_calls:,} prefix calls and 37,440 suffix calls remain in raw evidence.',
        f'All {len(blocks):,} consecutive 60-call blocks and all setups are published below,',
        'including the 144 measured suffix blocks. Exact rational clocks use equal',
        'process weights; no trimming, favorable block selection or duration search.', '',
        'The numerical oracle, shape, immutable-input and independently held-output',
        'checks pass. The public timer includes validation, materialization, output',
        'allocation, fill, mapping/copy and return. Setup and verification stay outside.',
        'Normal .NET 10.0.8 on AMD EPYC 9V74 CPU 2, monitoring CPU 0; no profiler,',
        'forced collection or implementation override. All owners are terminal/code zero.',
        f"All {analysis['resources']:,} resource samples pass; peak owned RSS {analysis['peak_rss']:,} bytes.", '',
        'These shapes use recorded frame counts and actual encoder pad widths, but',
        'are synthetic rather than captured intermediates. A passing screen permits',
        'full Parakeet/shared-model qualification; it does not establish application',
        'gain or ORT parity. The release remains 53.107381 versus ORT 39.207695 seconds.', '',
        '[All fixed blocks](screen-blocks-20260926.csv), [all setups](screen-setups-20260926.csv),',
        '[complete verdict and identities](screen-observations-20260926.json),',
        '[frozen prospective protocol](../pad-current-screen/README.md).', '',
        'Raw prefix and suffix clocks: artifacts/parakeet-pad-current-screen-amd-20260926/collected.',
        f"Closure: `{pin(BASE / 'closed.json')['sha256']}`."]
    with (OUT / 'screen-20260926.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(admitted=analysis['admitted'], prefix_calls=prefix_calls,
        suffix_calls=suffix_calls, blocks=len(blocks))))


if __name__ == '__main__': main()
