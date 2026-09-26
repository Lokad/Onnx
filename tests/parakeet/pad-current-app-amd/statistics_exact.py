"""All retained raw clocks, equal fresh-process weights, no cross-campaign pooling."""
from fractions import Fraction
from protocol import ROLES, TIMING_ROLES

def timing_table(results, manifest):
    assert len(results) == len(TIMING_ROLES) == 6
    assert len(manifest['cases']) == 20 and sum(c['samples'] for c in manifest['cases']) == 3412240
    def rational(value): return dict(numerator=value.numerator, denominator=value.denominator)
    table = []
    cases = manifest['cases']; assert len({c['name'] for c in cases}) == len(cases)
    for case in [*cases, dict(name='complete-corpus', samples=sum(c['samples'] for c in cases), is_corpus=True)]:
        full = case.get('is_corpus', False)
        names = {c['name'] for c in cases} if full else {case['name']}
        means = {}; row = dict(name=case['name'], audio_seconds=case['samples']/16000, is_corpus=full)
        for role in (*ROLES, 'ort'):
            processes = []; values = []
            for index, (assigned, result) in enumerate(zip(TIMING_ROLES, results, strict=True)):
                if role != assigned: continue
                by_case = {name: [Fraction(r['end_ticks']-r['start_ticks'], r['frequency']) for r in result['records']
                    if r['name'] == name and r['phase'] == 'measured'] for name in names}
                assert all(len(t) == 3 and all(v > 0 for v in t) for t in by_case.values())
                ticks = [sum(t[i] for t in by_case.values()) for i in range(3)]
                mean = sum(ticks)/3; values.extend(ticks)
                processes.append(dict(index=index, seconds=[float(t) for t in ticks], mean=float(mean), exact_mean=rational(mean)))
            assert len(values) == 6
            means[role] = sum(values)/6
            row[role] = dict(seconds=float(means[role]), exact_mean=rational(means[role]),
                rtf=float(means[role]/Fraction(case['samples'], 16000)),
                minimum=float(min(values)), maximum=float(max(values)), processes=processes)
        row['ratios_to_ort'] = {role: float(means[role]/means['ort']) for role in ROLES}
        table.append(row)
    return table

