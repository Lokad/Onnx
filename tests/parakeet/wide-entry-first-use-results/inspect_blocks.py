"""Describe all clocks in fixed 20-call blocks without changing the scored gate."""
import csv
import json
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/parakeet/wide-entry-first-use-screen'))
from protocol import pin, read
from score import ORDER

BASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-screen-amd-20260923'
OUT = Path(__file__).resolve().parent


def main():
    target = OUT / 'screen-blocks-20260923.csv'; assert not target.exists()
    closure = read(BASE / 'closed.json'); assert closure['passed']
    for name, wanted in closure['files'].items(): assert pin(BASE / name) == wanted, name
    rows = []
    for process in ORDER:
        result = read(BASE / 'collected' / process / 'result.json')
        for row in result['rows']:
            assert len(row['clocks']) == 120
            for start in range(0, 120, 20):
                clocks = row['clocks'][start:start + 20]
                assert [r['iteration'] for r in clocks] == list(range(start, start + 20))
                rows.append(dict(process=process, fixture=row['index'], m=row['m'], reduction=row['reduction'],
                    columns=row['columns'], first=start, last=start + 19, warmup=start < 60,
                    calls=20, ticks=sum(r['ticks'] for r in clocks), frequency=result['frequency']))
    assert len(rows) == 504 and sum(r['calls'] for r in rows) == 10080
    for row in rows: row['mean_ms'] = float(Fraction(row['ticks'] * 1000, row['calls'] * row['frequency']))
    with target.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(diagnostic_only=True, verdict_unchanged=True, admitted=closure['admitted'], closure=pin(BASE / 'closed.json'),
        blocks=len(rows)), indent=2))


if __name__ == '__main__': main()
