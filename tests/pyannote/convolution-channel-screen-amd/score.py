"""Exact-clock fixed gates; every full call contributes once per crop/pass."""
from fractions import Fraction

ORDER = ['production-a', 'candidate-a', 'candidate-b', 'production-b']


def iteration_manifest(calls):
    rows = []
    for call in calls:
        x, y, w = [call[key]['shape'] for key in ['input', 'output', 'weights']]
        assert len(x) == len(y) == len(w) == 4 and x[0] == y[0] == 1
        assert call['attributes']['group'] == 1 and w[:2] == [y[1], x[1]]
        assert all(type(v) is int and v > 0 for shape in [x, y, w] for v in shape)
        work = y[1]*x[1]*w[2]*w[3]*y[2]*y[3]
        count = max(1, ((1 << 31)+work-1)//work); assert count <= 128
        rows.append(dict(case=call['case'], index=call['index'], work=work, iterations=count))
    assert len({(r['case'], r['index']) for r in rows}) == len(rows)
    return dict(protocol='geometry-2pow31-v1', calls=rows, per_pass=sum(r['iterations'] for r in rows))


def call_totals(result, calls, expected, role):
    manifest = iteration_manifest(calls); per_pass = manifest['per_pass']
    assert result['protocol'] == manifest['protocol']
    assert result['cases'] == len(calls) and result['warmups'] == per_pass
    assert result['measured'] == 3*per_pass and result['calls'] == 4*per_pass
    assert len(result['observations']) == 4*per_pass
    totals = {c['form']: Fraction() for c in calls}; index = 0
    for repeat in range(4):
        for call, entry in zip(calls, manifest['calls'], strict=True):
            for iteration in range(entry['iterations']):
                row = result['observations'][index]; index += 1
                assert row['kind'] == 'call' and row['role'] == role and row['pass'] == repeat and row['warmup'] == (repeat == 0)
                assert (row['name'], row['index'], row['form'], row['eligible']) == (call['case'], call['index'], call['form'], call['eligible'])
                assert row['iteration'] == iteration and row['iterations'] == entry['iterations'] and row['work'] == entry['work']
                assert type(row['ticks']) is type(row['frequency']) is int and row['ticks'] > 0 and row['frequency'] > 0
                wanted = expected[(call['case'], call['index'])]
                assert row['exact'] and row['sha256'] == wanted['production'] and row['values'] == wanted['values']
                if repeat: totals[call['form']] += Fraction(row['ticks'], row['frequency'])/(3*entry['iterations'])
    assert index == len(result['observations'])
    return totals


def encode(value):
    return dict(numerator=value.numerator, denominator=value.denominator, seconds=float(value))


def evaluate(totals, eligible):
    assert list(totals) == ORDER
    forms = list(eligible); assert forms and all(set(totals[name]) == set(forms) for name in ORDER)
    assert all(isinstance(v, Fraction) and v > 0 for row in totals.values() for v in row.values())
    aggregate = {name: sum(row.values(), Fraction()) for name, row in totals.items()}
    controls = []
    for role in ['production', 'candidate']:
        names = [n for n in ORDER if n.startswith(role)]
        for form in [None, *forms]:
            values = [aggregate[n] if form is None else totals[n][form] for n in names]
            ratio = max(values)/min(values); limit = Fraction(11, 10) if form is None else Fraction(6, 5)
            controls.append(dict(role=role, form=form, ratio=encode(ratio), limit=encode(limit), passed=ratio <= limit))
    def mean(role, form):
        names = [n for n in ORDER if n.startswith(role)]
        return sum((aggregate[n] if form is None else totals[n][form] for n in names), Fraction())/len(names)
    rows = []; gates = []
    for form in [None, *forms]:
        p, c = mean('production', form), mean('candidate', form); ratio = c/p
        rows.append(dict(form=form, production=encode(p), candidate=encode(c), ratio=encode(ratio), eligible=form is None or eligible[form]))
        if form is None or eligible[form]:
            limit = Fraction(9, 10) if form is None else Fraction(21, 20)
            gates.append(dict(form=form, ratio=encode(ratio), limit=encode(limit), passed=ratio <= limit))
    return dict(admitted=all(r['passed'] for r in controls+gates), controls=controls, gates=gates, rows=rows,
        process_totals={name: dict(total=encode(aggregate[name]), forms={str(k): encode(v) for k, v in row.items()}) for name, row in totals.items()})


def validate_and_score(reports, fixtures, reference):
    assert list(reports) == ORDER
    calls = fixtures['calls']; assert len(calls) == 108
    expected = {(r['name'], r['index']): r for r in reference['observations']}; assert len(expected) == 108
    eligible = {}
    for call in calls:
        assert eligible.get(call['form'], call['eligible']) == call['eligible']; eligible[call['form']] = call['eligible']
    assert len(eligible) == 15
    constants = [c for c in calls[:36] if c['eligible']]; assert len(constants) == 32
    assert sum(c['weights']['bytes'] for c in constants) == 21086208
    totals = {}; preparations = {}
    for name, result in reports.items():
        role = name.split('-')[0]; assert result['passed'] and result['role'] == role and result['lanes'] == 16
        assert len(result['preparation']) == 128
        row_totals = call_totals(result, calls, expected, role); prep = Fraction()
        prep_hashes = {}
        for index, row in enumerate(result['preparation']):
            call = constants[index % 32]; repeat = index//32
            assert row['kind'] == 'preparation' and row['role'] == role and row['pass'] == repeat and row['warmup'] == (repeat == 0)
            assert row['index'] == call['index'] and row['node'] == call['node'] and row['bytes'] == call['weights']['bytes']
            assert type(row['ticks']) is type(row['frequency']) is int and row['ticks'] > 0 and row['frequency'] > 0
            assert prep_hashes.get(row['index'], row['sha256']) == row['sha256']; prep_hashes[row['index']] = row['sha256']
            if repeat: prep += Fraction(row['ticks'], row['frequency'])/3
        assert result['read_only_operands'] and result['prepared_bytes'] == 21086208
        totals[name] = row_totals; preparations[name] = encode(prep)
    result = evaluate(totals, eligible); result['preparation'] = preparations
    manifest = iteration_manifest(calls); result['iteration_manifest'] = manifest
    result['calls'] = 16*manifest['per_pass']; result['warmups'] = 4*manifest['per_pass']; result['measured'] = 12*manifest['per_pass']
    return result
