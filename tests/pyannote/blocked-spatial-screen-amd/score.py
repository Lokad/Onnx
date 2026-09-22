"""Exact-clock fixed gates; every full call contributes once per crop/pass."""
from fractions import Fraction

ORDER = ['production-a', 'candidate-a', 'candidate-b', 'production-b']


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
        assert result['cases'] == result['warmups'] == 108 and result['measured'] == 324 and result['calls'] == 432
        assert len(result['observations']) == 432 and len(result['preparation']) == 128
        row_totals = {form: Fraction() for form in eligible}; prep = Fraction()
        for index, row in enumerate(result['observations']):
            call = calls[index % 108]; repeat = index//108
            assert row['kind'] == 'call' and row['role'] == role and row['pass'] == repeat and row['warmup'] == (repeat == 0)
            assert (row['name'], row['index'], row['form'], row['eligible']) == (call['case'], call['index'], call['form'], call['eligible'])
            assert type(row['ticks']) is type(row['frequency']) is int and row['ticks'] > 0 and row['frequency'] > 0
            wanted = expected[(call['case'], call['index'])]
            assert row['exact'] and row['sha256'] == wanted['production'] and row['values'] == wanted['values']
            if repeat: row_totals[call['form']] += Fraction(row['ticks'], row['frequency'])/3
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
    result['calls'] = 1728; result['warmups'] = 432; result['measured'] = 1296
    return result
