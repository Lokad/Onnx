"""Fixed synthetic sensitivity checks, never a calibration of real VM behavior."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import statistics

from estimator import fieller, contained
from protocol import COHORTS, critical, PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scenarios = [dict(name='independent-process-sd-0.5pct', sigma=.005, phi=0),
                 dict(name='independent-process-sd-1pct', sigma=.01, phi=0),
                 dict(name='independent-process-sd-2pct', sigma=.02, phi=0),
                 dict(name='serial-process-sd-1pct', sigma=.01, phi=.85)]
    rows = []
    for scenario in scenarios:
        rng = random.Random('e5-prospective-sensitivity-fixed-v1-' + scenario['name'])
        sigma, phi = scenario['sigma'], scenario['phi']
        covered, passes, widths, ratios = 0, 0, [], []
        for trial in range(1000):
            x, y = [], []
            prior_x, prior_y = rng.gauss(0, sigma), rng.gauss(0, sigma)
            for cohort in range(COHORTS):
                common = rng.gauss(0, .03)
                prior_x = phi*prior_x + math.sqrt(1-phi**2)*rng.gauss(0, sigma)
                prior_y = phi*prior_y + math.sqrt(1-phi**2)*rng.gauss(0, sigma)
                x.append(round(10**9 * (1 + common + prior_x)))
                y.append(round(10**9 * (1 + common + prior_y)))
            interval = fieller(y, x, critical('aa'))
            if not interval['bounded']:
                raise ArithmeticError('unexpected unbounded synthetic interval')
            lo, hi = interval['interval']
            covers = lo <= 1 <= hi
            covered += covers
            passes += covers and contained(interval, .99, 1.01)
            widths.append((hi-lo)/2)
            ratios.append(interval['ratio'])
        rows.append(scenario | dict(trials=1000, covered_true_ratio=covered, aa_single_contrast_passes=passes,
                                    median_half_width=statistics.median(widths), maximum_half_width=max(widths),
                                    mean_observed_ratio=statistics.fmean(ratios),
                                    independent_cohorts=scenario['phi'] == 0))
        print(scenario['name'], rows[-1], flush=True)
    tools = {}
    for name in ('simulate.py', 'estimator.py', 'protocol.py'):
        path = Path(__file__).with_name(name)
        tools[name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = dict(protocol=PROTOCOL, source=tools, cohorts=COHORTS, family_contrasts=20,
                  individual_confidence=1-.05/20, critical=critical('aa'), true_ratio=1,
                  common_independent_cohort_sd=.03, results=rows,
                  scope='Synthetic marginal coverage and precision; not family-wide empirical coverage, power of the VM experiment, or evidence of real independence.')
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
