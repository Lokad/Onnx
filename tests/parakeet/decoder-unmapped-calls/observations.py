"""Reconcile individual intervals and test the predeclared first-call explanation."""
from fractions import Fraction as F

ORDER = ['current-0', 'candidate-1', 'candidate-2', 'current-3']


def number(value): return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def explain(positions, batches):
    assert list(positions) == list(batches) == ORDER
    assert all(len(row) == 7 and all(v > 0 for v in row) for row in positions.values())
    current = [(positions[ORDER[0]][i]+positions[ORDER[3]][i])/2 for i in range(7)]
    candidate = [(positions[ORDER[1]][i]+positions[ORDER[2]][i])/2 for i in range(7)]
    ratios = [b/a for a, b in zip(current, candidate, strict=True)]
    excess = [max(F(), b-a) for a, b in zip(current, candidate, strict=True)]
    concentration = excess[0]/sum(excess) if sum(excess) else F()
    batch_ratio = (batches[ORDER[1]]+batches[ORDER[2]])/(batches[ORDER[0]]+batches[ORDER[3]])
    checks = dict(batch_slowdown_reproduced=batch_ratio > F(21, 20), first_call_slower=ratios[0] > F(21, 20),
        all_later_positions_within_five_percent=all(r <= F(21, 20) for r in ratios[1:]),
        first_call_at_least_eighty_percent_of_positive_excess=concentration >= F(4, 5))
    return dict(first_call_explanation_supported=all(checks.values()), checks=checks, batch_ratio=number(batch_ratio),
        first_call_excess_fraction=number(concentration),
        positions=[dict(index=i, current_seconds=number(current[i]), candidate_seconds=number(candidate[i]),
            ratio=number(ratios[i]), positive_excess_seconds=number(excess[i])) for i in range(7)])


def analyze(reports):
    assert list(reports) == ORDER
    positions = {}; batches = {}; blocks = {}; warmups = {}; intervals = 0; overhead = {}
    for name, report in reports.items():
        assert report['diagnostic_only'] and not report['release_admitted']
        rows = [r for r in report['rows'] if r['kind'] == 'unmapped']; assert len(rows) == 1
        for row in report['rows']:
            if row['kind'] != 'unmapped': assert row['call_starts'] is None and row['call_ticks'] is None
        row = rows[0]; assert row['batch'] == 7 and len(row['clocks']) == 780
        starts, ticks = row['call_starts'], row['call_ticks']
        assert len(starts) == len(ticks) == 5460
        totals = [0]*7; warm = [0]*7; batch_ticks = 0; gap = 0
        for iteration, clock in enumerate(row['clocks']):
            end = clock['start']+clock['ticks']; previous = clock['start']; call_sum = 0
            for index in range(7):
                offset = iteration*7+index; start, duration = starts[offset], ticks[offset]
                assert type(start) is int and type(duration) is int and duration > 0
                assert start >= previous and start+duration <= end
                previous = start+duration; call_sum += duration; intervals += 1
                (warm if iteration < 600 else totals)[index] += duration
            assert call_sum <= clock['ticks']
            if iteration >= 600: batch_ticks += clock['ticks']; gap += clock['ticks']-call_sum
        frequency = report['frequency']; assert type(frequency) is int and frequency > 0
        positions[name] = [F(v, 180*frequency) for v in totals]
        warmups[name] = [number(F(v, 600*frequency)) for v in warm]
        batches[name] = F(batch_ticks, 180*frequency)
        overhead[name] = number(F(gap, 180*frequency))
        blocks[name] = [[number(F(sum(ticks[iteration*7+i] for iteration in range(start, start+60)), 60*frequency))
            for i in range(7)] for start in range(0, 780, 60)]
    assert intervals == 21840
    return dict(**explain(positions, batches), individual_intervals=intervals,
        measured_positions={name: [number(v) for v in row] for name, row in positions.items()},
        warmup_positions=warmups, sixty_round_positions=blocks, measured_outer_minus_inner_seconds=overhead,
        admitted=False, release_admitted=False, previous_screen_rescored=False)
