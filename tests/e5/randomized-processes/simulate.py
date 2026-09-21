"""Fixed potential-population scenarios, with retained assignments and intervals."""
from pathlib import Path
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import random
import statistics

from design import COHORTS, ORDERS, observed, population_ratio, fieller, student_critical, ESTIMATOR


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def scenarios():
    """Construct all potential tables before any assignment draws."""
    rng = random.Random('e5-fixed-potential-populations-v2')
    common, a_curve, c_curve = 0., 0., 0.
    rows = {name: [] for name in ('common-serial', 'common-trend-step', 'position-cost', 'role-serial',
                                 'heterogeneous-effect', 'dominant-unobserved-effect', 'carryover-counterexample')}
    for i in range(COHORTS):
        common = .95*common + math.sqrt(1-.95**2)*rng.gauss(0, .03)
        a_curve = .85*a_curve + math.sqrt(1-.85**2)*rng.gauss(0, .01)
        c_curve = .85*c_curve + math.sqrt(1-.85**2)*rng.gauss(0, .01)
        tables = {name: [] for name in rows}
        for position in range(3):
            noise = rng.gauss(0, .003)
            base = 1+common+noise
            tables['common-serial'].append([base, base, .9*base])
            trend = .9 + .1*i/(COHORTS-1) + (.07 if i >= 30 else 0) + .002*position + noise
            tables['common-trend-step'].append([trend, trend, .9*trend])
            positional = base + [0, .03, -.02][position]
            tables['position-cost'].append([positional, positional, .9*positional])
            tables['role-serial'].append([base+a_curve, base+c_curve, .9*base])
            tables['heterogeneous-effect'].append([base, base*(.97+.01*math.sin(i)), .9*base])
            tables['dominant-unobserved-effect'].append([base, base*(3 if i == 30 and position == 1 else 1), .9*base])
            tables['carryover-counterexample'].append([base, .97*base, .9*base])
        for name, table in tables.items():
            rows[name].append([[round(v*10**9) for v in slot] for slot in table])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=False, exist_ok=False)
    populations = scenarios()
    (base/'populations.json').write_text(json.dumps(populations, indent=2), encoding='utf8')
    critical = student_critical(1-.05/20, COHORTS-1)
    results = []
    with (base/'assignments.jsonl').open('x', encoding='utf8') as raw:
        for name, population in populations.items():
            target = population_ratio(population)
            rng = random.Random('e5-independent-assignments-v2-'+name)
            covers, passes, widths, ratios = 0, 0, [], []
            for trial in range(2000):
                draws = [rng.randrange(6) for _ in range(COHORTS)]
                x, y = [], []
                for table, draw in zip(population, draws, strict=True):
                    order = ORDERS[draw]
                    values = observed(table, order)
                    if name == 'carryover-counterexample':
                        # Deliberate interference: the immediately following
                        # worker is15%slower after C, breaking fixed potentials.
                        for position, role in enumerate(order):
                            if position and order[position-1] == 'C':
                                values[role] *= F(115, 100)
                    x.append(values['A']); y.append(values['C'])
                result = fieller(y, x, critical)
                assert result['bounded']
                lo, hi = result['interval']
                contains = lo <= float(target) <= hi
                covers += contains
                passes += lo <= 1 <= hi and .99 <= lo and hi <= 1.01
                widths.append((hi-lo)/2); ratios.append(result['ratio'])
                raw.write(json.dumps(dict(scenario=name, trial=trial, draws=draws, interval=[lo, hi],
                                          ratio=result['ratio'], contains_reference=contains))+'\n')
            row = dict(name=name, trials=2000, reference_ratio=str(target), reference_float=float(target),
                       intervals_containing_reference=covers, aa_single_contrast_passes=passes,
                       median_half_width=statistics.median(widths), mean_ratio=statistics.fmean(ratios),
                       no_interference=name != 'carryover-counterexample',
                       regularity_counterexample=name == 'dominant-unobserved-effect')
            results.append(row); print(json.dumps(row), flush=True)
    sources = {p.name: pin(p) for p in Path(__file__).parent.glob('*.py')}
    sources['../process-uncertainty/estimator.py'] = pin(ESTIMATOR)
    (base/'results.json').write_text(json.dumps(dict(critical=critical, marginal_confidence=1-.05/20,
        cohorts=COHORTS, results=results, source=sources,
        raw=dict(populations=pin(base/'populations.json'), assignments=pin(base/'assignments.jsonl')),
        scope='Fixed finite-population randomization sensitivity; no real VM observations or exact finite-sample coverage guarantee.'), indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
