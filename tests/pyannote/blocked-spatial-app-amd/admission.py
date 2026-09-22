"""Prospective portable-versus-root-production integration thresholds."""
from candidate_protocol import ROLES

ROLE_LABELS = dict(production='selected production (Core1279/Data4e60)',
    portable='prepared-convolution candidate (Core3c2f/Data6318)', ort='Microsoft ONNX Runtime 1.29.0')


def evaluate(table):
    assert len(table) == 4 and sorted(r['audio_seconds'] for r in table) == [10, 10, 10, 30]
    controls, gains = [], []
    for row in table:
        full = row['audio_seconds'] == 30
        for role in (*ROLES, 'ort'):
            means = [p['mean'] for p in row[role]['processes']]
            assert len(means) == 2 and min(means) > 0
            ratio = max(means) / min(means)
            limit = 1.10 if full else 1.20
            controls.append(dict(name=row['name'], role=role, process_ratio=ratio, limit=limit, passed=ratio <= limit))
        ratio = row['portable']['seconds'] / row['production']['seconds']
        limit = .97 if full else 1.05
        gains.append(dict(name=row['name'], control='production', ratio=ratio, limit=limit, passed=ratio <= limit))
    stable = all(r['passed'] for r in controls)
    gain = all(r['passed'] for r in gains)
    return dict(controls_passed=stable, speed_threshold_passed=gain, admitted=stable and gain,
        controls=controls, gains=gains, no_calibrated_parity_claim=True,
        policy='All roles: full process ratio <=1.10, crops <=1.20. Portable full mean <=0.97 of root production; crops <=1.05. Every sample retained; no unchanged retry.')
