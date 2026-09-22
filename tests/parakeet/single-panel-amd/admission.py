"""Fixed Parakeet process-repeatability and production-speed thresholds."""
from fractions import Fraction
from candidate_protocol import ROLES

ROLE_LABELS = dict(production='selected production (Core1279/Data4e60)',
    portable='Parakeet arithmetic candidate (Coreabbf/Dataeb45)', ort='Microsoft ONNX Runtime 1.29.0')


def exact(value):
    item = value['exact_mean']; result = Fraction(item['numerator'], item['denominator'])
    assert result > 0
    return result


def evaluate(table):
    assert len(table) == 21 and len({r['name'] for r in table}) == 21
    assert sum(r['is_corpus'] for r in table) == 1
    corpus = next(r for r in table if r['is_corpus'])
    assert corpus['audio_seconds'] == 213.265
    controls, gains = [], []
    for row in table:
        full = row['is_corpus']
        for role in (*ROLES, 'ort'):
            means = [exact(p) for p in row[role]['processes']]
            assert len(means) == 2 and exact(row[role]) == sum(means)/2
            ratio = max(means) / min(means)
            limit = Fraction(110 if full else 120, 100)
            controls.append(dict(name=row['name'], role=role, process_ratio=float(ratio), limit=float(limit), passed=ratio <= limit))
        ratio = exact(row['portable']) / exact(row['production'])
        limit = Fraction(95 if full else 105, 100)
        gains.append(dict(name=row['name'], control='production', ratio=float(ratio), limit=float(limit), passed=ratio <= limit))
    assert len(controls) == 63 and len(gains) == 21
    stable = all(r['passed'] for r in controls); gain = all(r['passed'] for r in gains)
    return dict(controls_passed=stable, speed_threshold_passed=gain, admitted=stable and gain,
        controls=controls, gains=gains, parity_target_met=exact(corpus['portable']) / exact(corpus['ort']) <= Fraction(105, 100),
        policy='All roles: corpus process ratio <=1.10, every clip <=1.20. Candidate corpus <=0.95 of production; every clip <=1.05. Exact integer-clock fractions; every sample retained; no unchanged retry.')
