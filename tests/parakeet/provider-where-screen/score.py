"""Exact-rational scoring; no clock deletion, minimum selection or post-hoc census."""
from fractions import Fraction as F
from protocol import ORDER


def number(value):
    return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def evaluate(totals, cases):
    assert list(totals) == ORDER and len(cases) == 122
    assert all(len(v) == 122 and all(x > 0 for x in v) for v in totals.values())
    groups = {'all122': list(range(122))}
    groups.update({scope: [i for i, c in enumerate(cases) if c['partition'] == scope]
                   for scope in ['target6', 'other_uniform42', 'fallback74']})
    assert [len(v) for v in groups.values()] == [122, 6, 42, 74]
    aggregates = {name: {scope: sum((values[i] for i in ids), F()) for scope, ids in groups.items()}
                  for name, values in totals.items()}
    controls = []; rows = []; scopes = {}
    for role, names in [('current', [ORDER[0], ORDER[3]]), ('candidate', ORDER[1:3])]:
        for scope in groups:
            v = [aggregates[name][scope] for name in names]; ratio = max(v) / min(v)
            controls.append(dict(role=role, scope=scope, ratio=number(ratio), limit=1.10, passed=ratio <= F(11, 10)))
        for i, case in enumerate(cases):
            v = [totals[name][i] for name in names]; ratio = max(v) / min(v)
            controls.append(dict(role=role, scope=case['name'], ratio=number(ratio), limit=1.20, passed=ratio <= F(6, 5)))
    for i, case in enumerate(cases):
        current = (totals[ORDER[0]][i] + totals[ORDER[3]][i]) / 2
        candidate = (totals[ORDER[1]][i] + totals[ORDER[2]][i]) / 2
        ratio = candidate / current
        rows.append(dict(index=i, name=case['name'], partition=case['partition'], current=number(current),
                         candidate=number(candidate), ratio=number(ratio), passed=ratio <= F(21, 20)))
    for scope in groups:
        current = (aggregates[ORDER[0]][scope] + aggregates[ORDER[3]][scope]) / 2
        candidate = (aggregates[ORDER[1]][scope] + aggregates[ORDER[2]][scope]) / 2
        scopes[scope] = dict(current=number(current), candidate=number(candidate), ratio=number(candidate/current))
    target = [aggregates[name]['target6'] for name in ORDER]
    gates = [dict(name='strict-target-process-sum-separation', passed=max(target[1:3]) < min(target[0], target[3])),
             dict(name='six-actual-targets-at-least-ten-percent', passed=target[1]+target[2] <= F(9, 10)*(target[0]+target[3])),
             dict(name='every-case-no-five-percent-regression', passed=all(r['passed'] for r in rows))]
    return dict(admitted=all(x['passed'] for x in [*controls, *gates]), controls=controls, gates=gates, rows=rows, scopes=scopes,
                processes={name: {scope: number(v) for scope, v in sums.items()} for name, sums in aggregates.items()})


def score(reports, cases):
    assert list(reports) == ORDER
    totals = {}; calls = 0
    for sequence, (name, value) in enumerate(reports.items()):
        assert value['completed'] and value['protocol'] == 'parakeet-provider-where-complete-call-60-60-v1'
        assert value['sequence'] == sequence and value['role'] == name.split('-')[0]
        frequency = value['frequency']; assert type(frequency) is int and frequency > 0
        assert len(value['rows']) == len(cases) == 122
        means = []
        for i, (row, case) in enumerate(zip(value['rows'], cases, strict=True)):
            assert row['index'] == i and row['name'] == case['name'] and row['dtype'] == case['dtype']
            assert all(row[k] for k in ['exact', 'inputs', 'owned', 'held'])
            assert row['output'] == case['expected_output'] and row['shape'] == case['output_shape']
            elements = 1
            for d in row['shape']: elements *= d
            batch = max(1, min(1024, 65536 // max(1, elements)))
            assert row['values'] == elements and row['batch'] == case['batch'] == batch
            assert len(row['clocks']) == 120
            measured = []
            for j, clock in enumerate(row['clocks']):
                assert clock['index'] == i and clock['name'] == case['name'] and clock['batch'] == batch
                assert clock['iteration'] == j and clock['warmup'] == (j < 60)
                assert type(clock['ticks']) is int and clock['ticks'] > 0
                if j >= 60: measured.append(F(clock['ticks'], frequency * batch))
            means.append(sum(measured, F()) / 60); calls += 120 * batch
        totals[name] = means
    return dict(**evaluate(totals, cases), sample_clocks=58560, warmup_clocks=29280, measured_clocks=29280,
                public_calls=calls, measured_public_calls=calls//2)
