"""Prospective descriptive selection gates, frozen before any target timing."""
from candidate_protocol import ROLES

ROLE_LABELS = dict(production='previous AVX-512 rows (Core294/Datae7)',
    portable='current portable-only (Coree9c/Data85)',
    rows='combined AVX-512 first (Coree369/Data2b51)', ort='Microsoft ONNX Runtime 1.29.0')


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
        for control in ('production', 'portable'):
            ratio = row['rows']['seconds'] / row[control]['seconds']
            limit = .97 if full else 1.05
            gains.append(dict(name=row['name'], control=control, ratio=ratio, limit=limit, passed=ratio <= limit))
    stable = all(r['passed'] for r in controls)
    gain = all(r['passed'] for r in gains)
    return dict(controls_passed=stable, speed_threshold_passed=gain, admitted=stable and gain,
        controls=controls, gains=gains, no_calibrated_parity_claim=True,
        policy='All roles: full process ratio <=1.10, crops <=1.20. Combined full mean <=0.97 of each managed control; crops <=1.05. No excluded sample or unchanged retry.')
