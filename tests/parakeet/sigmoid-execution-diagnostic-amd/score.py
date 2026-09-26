"""Reuse every record validator; report instrumented observations without admission."""
from fractions import Fraction
from scope import load, OLD

original = load('retained_sigmoid_record_validator', OLD / 'score.py')
ORDER = original.ORDER


def summarize(totals, cases):
    return dict(admitted=False, performance_admission_attempted=False,
                instrumented_case_seconds={name: [float(v) for v in rows] for name, rows in totals.items()})


def score(reports, census):
    # Keep the original public-result, shape, value, ownership and every-clock
    # checks. The final timing evaluator is deliberately diagnostic-only.
    original.evaluate = summarize
    value = original.score(reports, census)
    observations = {}
    for name, report in reports.items():
        assert report['diagnostic_only'] and report['vector_width'] in [4, 8, 16]
        rows = []
        for case, row in zip(census['cases'], report['rows'], strict=True):
            assert len(row['observations']) == 780
            selected = []
            for index, record in enumerate(row['observations']):
                assert record['iteration'] == index
                assert all(type(record[k]) is int and record[k] >= 0 for k in ['allocated', 'gen0', 'gen1', 'gen2'])
                if index >= 600:
                    selected.append(record)
            allocations = [Fraction(r['allocated'], case['batch']) for r in selected]
            rows.append(dict(name=case['name'], batch=case['batch'], elements=case['elements'],
                minimum_allocated_bytes=float(min(allocations)), maximum_allocated_bytes=float(max(allocations)),
                mean_allocated_bytes=float(sum(allocations) / 180),
                collections={k:sum(r[k] for r in selected) for k in ['gen0', 'gen1', 'gen2']},
                batches_with_collection=sum(any(r[k] for k in ['gen0', 'gen1', 'gen2']) for r in selected)))
        observations[name] = dict(vector_width=report['vector_width'], cases=rows)
    return dict(**value, observations=observations, no_clock_trimmed=True,
                allocation_counters_outside_timer=True, arithmetic_cost_not_directly_sampled=True)
