"""Recompute every simulated ratio/interval from raw assignments using Decimal."""
from pathlib import Path
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import itertools
import json
import random
import statistics


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args()
    base = args.artifact.resolve()
    target_file = base/'simulation-verification.json'
    assert not target_file.exists()
    meta = json.loads((base/'results.json').read_text())
    for name, value in meta['source'].items():
        assert pin(Path(__file__).parent/name) == value, name
    for name in ('populations', 'assignments'):
        assert pin(base/(name+('.json' if name == 'populations' else '.jsonl'))) == meta['raw'][name]
    populations = json.loads((base/'populations.json').read_text())
    roles = ('A', 'C', 'N')
    orders = list(itertools.permutations(roles))
    expected = {r['name']: r for r in meta['results']}
    aggregate = {name: dict(trials=0, covered=0, passed=0, widths=[], ratios=[]) for name in populations}
    randomizers = {name: random.Random('e5-independent-assignments-v2-'+name) for name in populations}
    maximum_error = 0.
    with localcontext() as ctx:
        ctx.prec = 60
        q = Decimal.from_float(meta['critical'])**2
        with (base/'assignments.jsonl').open() as raw:
            for line in raw:
                row = json.loads(line); name = row['scenario']; table = populations[name]
                assert row['trial'] == aggregate[name]['trials']
                assert row['draws'] == [randomizers[name].randrange(6) for _ in range(60)]
                x, y = [], []
                for potentials, draw in zip(table, row['draws'], strict=True):
                    order = orders[draw]
                    values = {role: Decimal(potentials[slot][roles.index(role)]) for slot, role in enumerate(order)}
                    if name == 'carryover-counterexample':
                        for slot in (1, 2):
                            if order[slot-1] == 'C':
                                values[order[slot]] *= Decimal('1.15')
                    x.append(values['A']); y.append(values['C'])
                n = Decimal(len(x)); sx, sy = sum(x), sum(y)
                mx, my = sx/n, sy/n
                vx = (sum(v*v for v in x)-sx*sx/n)/(n*(n-1))
                vy = (sum(v*v for v in y)-sy*sy/n)/(n*(n-1))
                cov = (sum(a*b for a, b in zip(x, y, strict=True))-sx*sy/n)/(n*(n-1))
                a, b, c = mx*mx-q*vx, mx*my-q*cov, my*my-q*vy
                assert a > 0 and b*b-a*c >= 0
                radius = (b*b-a*c).sqrt()
                bounds = [float((b-radius)/a), float((b+radius)/a)]
                ratio = float(my/mx)
                for got, want in zip(row['interval']+[row['ratio']], bounds+[ratio], strict=True):
                    error = abs(got-want); maximum_error = max(maximum_error, error)
                    assert error < 1e-14
                reference = Fraction(sum(slot[1] for cohort in table for slot in cohort),
                                     sum(slot[0] for cohort in table for slot in cohort))
                assert str(reference) == expected[name]['reference_ratio']
                lo, hi = row['interval']
                contains = lo <= float(reference) <= hi
                assert row['contains_reference'] is contains
                current = aggregate[name]
                current['trials'] += 1; current['covered'] += contains
                current['passed'] += lo <= 1 <= hi and .99 <= lo and hi <= 1.01
                current['widths'].append((hi-lo)/2); current['ratios'].append(row['ratio'])
    for name, actual in aggregate.items():
        want = expected[name]
        assert actual['trials'] == want['trials'] == 2000
        assert actual['covered'] == want['intervals_containing_reference']
        assert actual['passed'] == want['aa_single_contrast_passes']
        assert statistics.median(actual['widths']) == want['median_half_width']
        assert statistics.fmean(actual['ratios']) == want['mean_ratio']
    output = dict(passed=True, trials=sum(r['trials'] for r in aggregate.values()),
                  independently_recomputed_endpoints=42000, maximum_absolute_difference=maximum_error,
                  decimal_precision=60, raw={p.name: pin(p) for p in base.iterdir() if p.is_file()},
                  verifier=pin(Path(__file__)), scope='Arithmetic and evidence verification only; no VM qualification.')
    with target_file.open('x', encoding='utf8') as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k not in ('raw', 'verifier')}))


if __name__ == '__main__':
    main()
