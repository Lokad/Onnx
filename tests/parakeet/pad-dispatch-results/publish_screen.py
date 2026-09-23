"""Publish every fixed Pad clock and the unchanged prospective verdict."""
import csv
from fractions import Fraction as F
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pad-dispatch-screen-amd-20260923'
sys.path.insert(0, str(ROOT / 'tests/parakeet/last-axis-pad-screen'))
from protocol import pin, read
from score import ORDER, score


def csv_file(name, rows):
    with (OUT / name).open('x', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    proof = read(BASE / 'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(BASE / name) == wanted, name
    analysis = read(BASE / 'analysis.json')
    reports = {name: read(BASE / 'collected' / name / 'result.json') for name in ORDER}
    for key, value in score(reports).items(): assert analysis[key] == value, key
    clocks, setups, blocks = [], [], []
    for process, result in reports.items():
        frequency = result['frequency']
        for row in result['rows']:
            setups.append(dict(process=process, case=row['name'], ticks=row['setupTicks'], frequency=frequency))
            for c in row['clocks']:
                clocks.append(dict(process=process, case=row['name'], **c, frequency=frequency))
            for start in [600, 660, 720]:
                mean = sum((F(c['ticks'], frequency) for c in row['clocks'][start:start+60]), F())/60
                blocks.append(dict(process=process, case=row['name'], first=start, last=start+59,
                                   numerator=mean.numerator, denominator=mean.denominator, seconds=float(mean)))
    assert (len(clocks), len(setups), len(blocks)) == (37440, 48, 144)
    csv_file('clocks-20260923.csv', clocks)
    csv_file('setup-20260923.csv', setups)
    csv_file('blocks-20260923.csv', blocks)
    with (OUT / 'screen-observations-20260923.json').open('x', encoding='utf8') as f:
        json.dump(dict(closure=pin(BASE / 'closed.json'), **analysis), f, indent=2); f.write('\n')
    headline = '**The complete Pad screen passes its fixed gates.**' if proof['admitted'] else '**The complete Pad screen does not pass its fixed gates; no model trial is admitted.**'
    lines = ['# Parakeet complete public Pad comparison', '', headline, '',
             '| Synthetic case | Selected ms | Candidate ms | Candidate / selected | Regression check |',
             '| --- | ---: | ---: | ---: | --- |']
    for row in analysis['rows']:
        lines.append(f"| {row['name']} | {row['current']['value']*1000:.6f} | {row['candidate']['value']*1000:.6f} | {row['ratio']['value']:.6f} | {'Pass' if row['passed'] else 'Fail'} |")
    ratio = analysis['eligible']['ratio']['value']
    lines += ['', f"The eight primary case means improve by {(1-ratio)*100:.6f}%.",
              f"Repeatability: {sum(c['passed'] for c in analysis['controls'])}/26 controls pass (limit 1.10).",
              f"All-case regression: {sum(r['passed'] for r in analysis['rows'])}/12 pass (limit 1.05).", '']
    for gate in analysis['gates']:
        lines.append(f"- {gate['name']}: {'pass' if gate['passed'] else 'fail'}.")
    lines += ['', 'AMD EPYC 9V74, CPU 2 with CPU 0 monitoring; normal .NET 10.0.8.',
              'Four fresh selected/candidate/candidate/selected processes, twelve cases,',
              '600 fixed warmups and 180 measurements each. All 37,440 clocks, 8,640',
              'measurements, 48 setups and 144 fixed measured blocks remain. Exact rational',
              'clocks give equal process weights; no trimming, block selection or retry.', '',
              'Timing includes the complete public Pad operation: validation, materialization,',
              'allocation, fill and mapping/copy. Oracle/output checks and ownership mutations',
              'are outside the interval. All output bits, shapes, unchanged inputs and',
              'independently held outputs pass. Both products use the same consumer.', '',
              'Primary shapes are synthetic attention [1,8,T,2T-1] and convolution [1,1024,T]',
              'for recorded frame counts T=51/106/167/225. Pad widths match the actual',
              'encoder graph; these are not captured Pad intermediates. Coverage includes',
              'zero padding and unchanged crop/outer-pad/reflection paths. This result',
              'does not establish a full Parakeet application gain or Microsoft ORT parity.', '',
              f"All owners are terminal; {analysis['resources']:,} resource samples pass, peak RSS {analysis['peak_rss']:,} bytes.", '',
              '[All clocks](clocks-20260923.csv), [setups](setup-20260923.csv),',
              '[fixed blocks](blocks-20260923.csv), [all gates and identities](screen-observations-20260923.json),',
              '[prospective protocol](../pad-dispatch-screen/README.md).', '',
              'Closure: `' + pin(BASE / 'closed.json')['sha256'] + '`.']
    with (OUT / 'screen-20260923.md').open('x', encoding='utf8') as f: f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(admitted=proof['admitted'], calls=len(clocks), measured=analysis['measured'])))


if __name__ == '__main__':
    main()
