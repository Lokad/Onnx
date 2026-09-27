"""Complete untrimmed clocks; fixed target, repeatability and fallback gates."""
from fractions import Fraction as F

ORDER = ['current-0', 'candidate-1', 'candidate-2', 'current-3']


def number(value): return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def evaluate(totals, cases):
    assert list(totals) == ORDER and len(cases) == 6
    assert all(len(v) == 6 and all(t > 0 for t in v) for v in totals.values())
    controls = []; rows = []
    for role, names in [('current', [ORDER[0], ORDER[3]]), ('candidate', ORDER[1:3])]:
        for i, case in enumerate(cases):
            values = [totals[n][i] for n in names]; ratio = max(values)/min(values)
            controls.append(dict(role=role, case=case['name'], ratio=number(ratio), passed=ratio <= F(11, 10)))
    for i, case in enumerate(cases):
        current = (totals[ORDER[0]][i]+totals[ORDER[3]][i])/2
        candidate = (totals[ORDER[1]][i]+totals[ORDER[2]][i])/2
        limit = F(3, 4) if case['kind'] == 'target' else F(21, 20)
        rows.append(dict(index=i, name=case['name'], current=number(current), candidate=number(candidate),
            ratio=number(candidate/current), limit=number(limit), passed=candidate/current <= limit))
    return dict(admitted=all(v['passed'] for v in controls+rows), controls=controls, rows=rows,
        target=rows[0], processes={n: [number(v) for v in ts] for n, ts in totals.items()})


def score(reports, census):
    assert list(reports) == ORDER
    cases = census['cases']; assert len(cases) == 6 and cases[0]['kind'] == 'target'
    batches = sum(c['batch'] for c in cases); totals = {}; hashes = []; counters = {}; checked = 0
    for sequence, (name, report) in enumerate(reports.items()):
        assert report['passed'] and report['protocol'] == census['protocol'] == 'decoder-packed-row-public-600-180-v1'
        assert report['role'] == name.split('-')[0] and report['sequence'] == sequence
        assert (report['samples'], report['warmup_samples'], report['measured_samples']) == (4680, 3600, 1080)
        assert (report['calls'], report['warmups'], report['measured']) == (780*batches, 600*batches, 180*batches)
        assert len(report['rows']) == 6 and type(report['frequency']) is int and report['frequency'] > 0
        means = []; identities = []; accounting = []
        for i, (row, case) in enumerate(zip(report['rows'], cases, strict=True)):
            assert row['index'] == i and all(row[k] == case[k] for k in ['name', 'kind', 'shape', 'batch'])
            assert row['inputs'] and row['ownership'] and row['exact'] and row['setup_ticks'] > 0
            identities.append([row[k] for k in ['input_sha256', 'weight_sha256', 'output_sha256']])
            assert all(len(h) == 64 for h in identities[-1])
            assert len(row['clocks']) == 780
            ticks = 0; values = {k: 0 for k in ['allocated', 'copies', 'scratch']}
            warm = dict(values)
            for iteration, clock in enumerate(row['clocks']):
                assert clock['iteration'] == iteration and clock['warmup'] == (iteration < 600)
                assert type(clock['start']) is int and clock['start'] > 0 and type(clock['ticks']) is int and clock['ticks'] > 0
                for k in values:
                    assert type(clock[k]) is int and clock[k] >= 0
                    (warm if iteration < 600 else values)[k] += clock[k]
                if iteration >= 600: ticks += clock['ticks']
            means.append(F(ticks, report['frequency']*180*case['batch']))
            accounting.append(dict(name=case['name'], warmup_totals=warm, measured_totals=values,
                measured_bytes_per_call={k: number(F(v, 180*case['batch'])) for k, v in values.items()}))
            checked += 3*case['batch']
        previous = 0
        for iteration in range(780):
            for row in report['rows']:
                clock = row['clocks'][iteration]
                assert clock['start'] >= previous; previous = clock['start']+clock['ticks']
        totals[name] = means; hashes.append(identities); counters[name] = accounting
    assert all(h == hashes[0] for h in hashes), 'Every output exact across products'
    verdict = evaluate(totals, cases)
    gates = []
    for i, case in enumerate(cases):
        for phase in ['warmup_totals', 'measured_totals']:
            for kind in ['copies', 'scratch']:
                current = sum(counters[n][i][phase][kind] for n in [ORDER[0], ORDER[3]])
                candidate = sum(counters[n][i][phase][kind] for n in ORDER[1:3])
                gates.append(dict(case=case['name'], phase=phase, kind=kind, current=current, candidate=candidate, passed=candidate <= current))
    verdict['admitted'] = verdict['admitted'] and all(g['passed'] for g in gates)
    return dict(**verdict, accounting=counters, gates=gates, samples=18720, measured_samples=4320, warmup_samples=14400,
        calls=3120*batches, measured_calls=720*batches, warmup_calls=2400*batches, setups=24,
        exact_output_checks=checked, identities=hashes[0], all_warmups_precede_measures=True)
