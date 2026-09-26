"""Exact complete-call means; retain all clocks and every failure."""
from fractions import Fraction as F
from census import census

ORDER = ['current-screen0-512', 'candidate-screen1-512', 'candidate-screen2-512', 'current-screen3-512']


def number(value):
    return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def evaluate(totals):
    assert list(totals) == ORDER
    assert all(len(v) == 12 and all(x > 0 for x in v) for v in totals.values())
    controls, rows = [], []
    eligible = {key: sum(values[:8], F()) for key, values in totals.items()}
    for role, names in [('current', [ORDER[0], ORDER[3]]), ('candidate', ORDER[1:3])]:
        for i in range(12):
            values = [totals[name][i] for name in names]
            ratio = max(values)/min(values)
            controls.append(dict(role=role, case=i, ratio=number(ratio), passed=ratio <= F(11, 10)))
        ratio = max(eligible[name] for name in names)/min(eligible[name] for name in names)
        controls.append(dict(role=role, case='eligible-sum', ratio=number(ratio), passed=ratio <= F(11, 10)))
    for i in range(12):
        current = (totals[ORDER[0]][i] + totals[ORDER[3]][i])/2
        candidate = (totals[ORDER[1]][i] + totals[ORDER[2]][i])/2
        ratio = candidate/current
        rows.append(dict(index=i, name=census()['cases'][i]['name'], current=number(current),
                         candidate=number(candidate), ratio=number(ratio), passed=ratio <= F(21, 20)))
    current = (eligible[ORDER[0]] + eligible[ORDER[3]])/2
    candidate = (eligible[ORDER[1]] + eligible[ORDER[2]])/2
    gates = [dict(name='eligible-at-least-twenty-percent', passed=candidate/current <= F(4, 5)),
             dict(name='strict-process-sum-separation', passed=max(eligible[n] for n in ORDER[1:3]) < min(eligible[n] for n in [ORDER[0], ORDER[3]])),
             dict(name='all-twelve-no-five-percent-regression', passed=all(r['passed'] for r in rows))]
    return dict(admitted=all(v['passed'] for v in controls + gates), controls=controls, gates=gates, rows=rows,
                eligible=dict(current=number(current), candidate=number(candidate), ratio=number(candidate/current)),
                processes={name: number(value) for name, value in eligible.items()})


def score(reports):
    assert list(reports) == ORDER
    totals, outputs = {}, []
    for sequence, (name, value) in enumerate(reports.items()):
        assert value['passed'] and value['protocol'] == 'parakeet-pad-public-600-180-v1'
        assert value['role'] == name.split('-')[0] and value['sequence'] == sequence
        assert (value['calls'], value['warmups'], value['measured']) == (9360, 7200, 2160)
        assert len(value['rows']) == 12 and type(value['frequency']) is int and value['frequency'] > 0
        means, hashes = [], []
        for i, (row, case) in enumerate(zip(value['rows'], census()['cases'], strict=True)):
            assert row['index'] == i
            assert all(row[k] == case[k] for k in ['name', 'shape', 'pads', 'mode', 'fill'])
            assert row['exact'] and row['inputs'] and row['ownership']
            assert type(row['setupTicks']) is int and row['setupTicks'] > 0
            assert len(row['output']) == 64 and len(row['clocks']) == 780
            hashes.append(row['output']); measured = []
            for j, clock in enumerate(row['clocks']):
                assert type(clock['iteration']) is int and clock['iteration'] == j
                assert type(clock['warmup']) is bool and clock['warmup'] == (j < 600)
                assert type(clock['ticks']) is int and clock['ticks'] > 0
                if not clock['warmup']:
                    measured.append(F(clock['ticks'], value['frequency']))
            means.append(sum(measured, F())/180)
        totals[name] = means; outputs.append(hashes)
    assert all(values == outputs[0] for values in outputs)
    return dict(**evaluate(totals), calls=37440, warmups=28800, measured=8640, setups=48,
                complete_output_hashes_equal=True)
