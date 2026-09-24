"""Exact-rational identical-binary controls; every measured clock is included."""
from fractions import Fraction as F
from protocol import ORDER


def number(value):
    return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def evaluate(totals, cases):
    assert list(totals) == ORDER and len(cases) == 220
    assert all(len(v) == 220 and all(x > 0 for x in v) for v in totals.values())
    groups = {'all220': list(range(220))}
    groups.update({scope: [i for i, c in enumerate(cases) if c['partition'] == scope]
                   for scope in ['target6', 'other_uniform42', 'fallback74', 'added98']})
    assert [len(v) for v in groups.values()] == [220, 6, 42, 74, 98]
    aggregates = {name: {scope: sum((values[i] for i in ids), F()) for scope, ids in groups.items()}
                  for name, values in totals.items()}
    controls = []; rows = []; scopes = {}
    for scope in groups:
        values = [aggregates[name][scope] for name in ORDER]; ratio = max(values) / min(values)
        controls.append(dict(scope=scope, ratio=number(ratio), limit=1.10, passed=ratio <= F(11,10)))
    for i, case in enumerate(cases):
        values = [totals[name][i] for name in ORDER]; ratio = max(values) / min(values)
        controls.append(dict(scope=case['name'], ratio=number(ratio), limit=1.20, passed=ratio <= F(6,5)))
        outer = (totals[ORDER[0]][i] + totals[ORDER[3]][i]) / 2
        middle = (totals[ORDER[1]][i] + totals[ORDER[2]][i]) / 2
        ratio = middle / outer
        rows.append(dict(index=i, name=case['name'], partition=case['partition'], outer=number(outer),
                         middle=number(middle), ratio=number(ratio), passed=F(20,21) <= ratio <= F(21,20)))
    for scope in groups:
        outer = (aggregates[ORDER[0]][scope] + aggregates[ORDER[3]][scope]) / 2
        middle = (aggregates[ORDER[1]][scope] + aggregates[ORDER[2]][scope]) / 2
        ratio = middle / outer
        scopes[scope] = dict(outer=number(outer), middle=number(middle), ratio=number(ratio), passed=F(20,21) <= ratio <= F(21,20))
    gates = [dict(name='every-identical-case-within-symmetric-five-percent', passed=all(r['passed'] for r in rows)),
             dict(name='every-identical-aggregate-within-symmetric-five-percent', passed=all(r['passed'] for r in scopes.values()))]
    return dict(admitted=all(x['passed'] for x in [*controls, *gates]), controls=controls, gates=gates, rows=rows, scopes=scopes,
                processes={name: {scope: number(v) for scope, v in sums.items()} for name, sums in aggregates.items()})


def score(reports, cases):
    assert list(reports) == ORDER
    totals = {}; calls = 0
    for sequence, (name, value) in enumerate(reports.items()):
        assert value['completed'] and value['protocol'] == 'parakeet-dense-where-whole-census-600-180-v1'
        assert value['sequence'] == sequence and value['role'] == name.split('-')[0]
        frequency = value['frequency']; assert type(frequency) is int and frequency > 0
        assert len(value['rows']) == len(cases) == 220
        means = []
        for i, (row, case) in enumerate(zip(value['rows'], cases, strict=True)):
            assert row['index'] == i and row['name'] == case['name'] and row['dtype'] == case['dtype']
            assert all(row[k] for k in ['exact', 'inputs', 'owned', 'held'])
            assert row['output'] == case['expected_output'] and row['shape'] == case['output_shape']
            elements = 1
            for d in row['shape']: elements *= d
            batch = max(1, min(1024, 65536 // max(1, elements)))
            assert row['values'] == elements and row['batch'] == case['batch'] == batch
            assert len(row['clocks']) == 780
            measured = []
            for j, clock in enumerate(row['clocks']):
                assert clock['index'] == i and clock['name'] == case['name'] and clock['batch'] == batch
                assert clock['iteration'] == j and clock['warmup'] == (j < 600)
                assert type(clock['ticks']) is int and clock['ticks'] > 0
                if j >= 600: measured.append(F(clock['ticks'], frequency * batch))
            means.append(sum(measured, F()) / 180); calls += 780 * batch
        totals[name] = means
    return dict(**evaluate(totals, cases), sample_clocks=686400, warmup_clocks=528000, measured_clocks=158400,
                public_calls=calls, measured_public_calls=calls*180//780, identical_binary_control=True)
