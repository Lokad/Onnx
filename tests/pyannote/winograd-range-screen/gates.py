from fractions import Fraction
ORDER=['current-a','candidate-a','candidate-b','current-b']

def encode(value):
    return dict(numerator=value.numerator, denominator=value.denominator, seconds=float(value))


def evaluate(totals, eligible):
    assert list(totals) == ORDER
    forms = list(eligible); assert forms and all(set(totals[name]) == set(forms) for name in ORDER)
    assert all(isinstance(v, Fraction) and v > 0 for row in totals.values() for v in row.values())
    aggregate = {name: sum(row.values(), Fraction()) for name, row in totals.items()}
    controls = []
    for role in ['current', 'candidate']:
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
        p, c = mean('current', form), mean('candidate', form); ratio = c/p
        rows.append(dict(form=form, current=encode(p), candidate=encode(c), ratio=encode(ratio), eligible=form is None or eligible[form]))
        if form is None or eligible[form]:
            limit = Fraction(9, 10) if form is None else Fraction(21, 20)
            gates.append(dict(form=form, ratio=encode(ratio), limit=encode(limit), passed=ratio <= limit))
    candidate_max=max(aggregate[n] for n in ORDER if n.startswith('candidate'))
    current_min=min(aggregate[n] for n in ORDER if n.startswith('current'))
    separation=dict(candidate_max=encode(candidate_max),current_min=encode(current_min),
                    passed=candidate_max<current_min)
    return dict(admitted=all(r['passed'] for r in controls+gates) and separation['passed'],
        controls=controls, gates=gates, rows=rows, process_separation=separation,
        process_totals={name: dict(total=encode(aggregate[name]), forms={str(k): encode(v) for k, v in row.items()}) for name, row in totals.items()})

