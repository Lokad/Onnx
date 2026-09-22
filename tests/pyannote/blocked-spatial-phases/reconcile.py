"""Independent inclusive clock reconciliation, coverage and diagnostic aggregation."""
from fractions import Fraction

PHASES = ['allocation', 'boundaries', 'validation', 'input_layout', 'arithmetic',
          'output_epilogue', 'pool_return', 'wrapper', 'fallback']


def interval(row, frequency):
    assert type(frequency) is int and frequency > 0 and row['frequency'] == frequency
    assert all(type(row[k]) is int for k in ['start', 'stop', 'before', 'after', 'thread'])
    assert row['start'] >= 0 and row['stop'] > row['start'] and row['thread'] > 0
    result = {name: 0 for name in PHASES}
    if row['eligible']:
        t = row['inner']; assert len(t) == 9 and all(type(v) is int for v in t)
        assert row['after'] == row['before'] + 1
        chain = [row['start'], *t, row['stop']]
        assert all(b >= a for a, b in zip(chain, chain[1:]))
        differences = [b-a for a, b in zip(chain, chain[1:])]
        result.update(allocation=differences[1], boundaries=differences[2]+differences[7],
            validation=differences[3], input_layout=differences[4], arithmetic=differences[5],
            output_epilogue=differences[6], pool_return=differences[8], wrapper=differences[0]+differences[9])
    else:
        assert row['inner'] == [] and row['after'] == row['before']
        result['fallback'] = row['stop']-row['start']
    assert row['phases'] == result and all(type(v) is int and v >= 0 for v in row['phases'].values())
    assert sum(result.values()) == row['stop']-row['start']
    return result


def records(rows, calls, expected, frequency):
    assert len(calls) == len(expected) == 108 and len(rows) == 432
    assert len({(r['case'], r['index']) for r in calls}) == 108
    assert len({r['thread'] for r in rows}) == 1
    sequence = 0; total = {name: Fraction() for name in PHASES}; warm = dict(total)
    forms = {r['form']: dict(total) for r in calls}; eligible = 0
    for index, row in enumerate(rows):
        call = calls[index % 108]; repeat = index//108
        assert row['pass'] == repeat and row['warmup'] == (repeat == 0)
        assert (row['name'], row['index'], row['form'], row['eligible']) == (call['case'], call['index'], call['form'], call['eligible'])
        wanted = expected[(call['case'], call['index'])]
        assert row['exact'] and row['sha256'] == wanted['production'] and row['values'] == wanted['values']
        assert row['before'] == sequence
        phase = interval(row, frequency); sequence = row['after']; eligible += row['eligible']
        for name, ticks in phase.items():
            if repeat:
                value = Fraction(ticks, frequency)/3; total[name] += value; forms[call['form']][name] += value
            else: warm[name] += Fraction(ticks, frequency)
    assert eligible == sequence == 384
    for name in PHASES: assert total[name] == sum((r[name] for r in forms.values()), Fraction())
    return dict(total=total, warmup=warm, forms=forms, calls=len(rows), eligible=eligible, fallback=len(rows)-eligible)


def encode(value): return dict(numerator=value.numerator, denominator=value.denominator, seconds=float(value))


def table(phases):
    total = sum(phases.values(), Fraction()); assert total > 0
    return dict(total=encode(total), phases={k: dict(**encode(v), share=float(v/total)) for k, v in phases.items()})


def analyze(reports, fixtures, reference):
    assert list(reports) == ['profile-a', 'profile-b']
    expected = {(r['name'], r['index']): r for r in reference['observations']}
    results = {}
    for name, report in reports.items():
        assert report['passed'] and report['instrumented_diagnostic_only'] and report['read_only_operands']
        assert report['cases'] == report['warmups'] == 108 and report['measured'] == 324 and report['calls'] == 432
        assert report['prepared_bytes'] == 21086208 and report['frequency'] == 1000000000
        value = records(report['rows'], fixtures['calls'], expected, report['frequency'])
        assert len(value['forms']) == 15
        results[name] = dict(total=table(value['total']), warmup=table(value['warmup']),
            forms={str(k): table(v) for k, v in value['forms'].items()}, calls=value['calls'], eligible=value['eligible'], fallback=value['fallback'])
    totals = [Fraction(r['total']['total']['numerator'], r['total']['total']['denominator']) for r in results.values()]
    return dict(passed=True, calls=864, eligible=768, fallback=96, processes=results,
        diagnostic_process_max_min=encode(max(totals)/min(totals)), no_speed_selection=True, no_application_or_ort_timing=True)
