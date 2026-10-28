"""Full audio result checks shared by the native worker and its supervisor."""
import hashlib
import json
import math
from pathlib import Path


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def check_result(actual, expected, path='', family=None):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        return max([check_result(actual[k], v, path+'/'+k, family) for k, v in expected.items()] + [0.])
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        return max([check_result(a, e, path+'/'+str(i), family) for i, (a, e) in enumerate(zip(actual, expected))] + [0.])
    if type(expected) in (int, float):
        assert type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected), path
        if family == 'parakeet' and type(expected) is int:
            assert type(actual) is int, path
        if '/centroid/' in path:
            error = abs(actual-expected)/max(1, abs(expected))
            assert error <= 1e-4, path
            return error
        tolerance = 1e-12 if path.startswith(('/intervals/', '/exclusive_intervals/')) else 0
        assert abs(actual-expected) <= tolerance, path
    else:
        assert type(actual) is type(expected) and actual == expected, path
    return 0.


def validate_records(value, manifest, mode):
    assert value['schema'] == 1 and value['family'] == manifest['family'] and value['engine'] in ('managed', 'ort')
    assert value['conformance'] == (mode == 'conformance') and value['held_outputs_unchanged'] is True and value['affinity'] == 4
    assert math.isfinite(value['setup_seconds']) and value['setup_seconds'] >= 0
    assert not any(k.lower().startswith(('lokad_', 'dotnet_', 'complus_')) for k in value['flags'])
    warmups = manifest['warmup_passes']
    wanted = [(iteration, case) for iteration in range(1 if mode == 'conformance' else warmups+manifest['measured_passes']) for case in manifest['cases']]
    assert len(value['records']) == len(wanted)
    previous, first = 0, {}
    for row, (iteration, case) in zip(value['records'], wanted):
        assert row['name'] == case['name'] and row['pass'] == iteration
        assert row['phase'] == ('warmup' if mode == 'conformance' or iteration < warmups else 'measured')
        assert row['ownership'] is True and row['input_sha256'] == case['raw_sha256']
        assert all(type(row[k]) is int for k in ['start_ticks', 'end_ticks', 'frequency']) and row['frequency'] > 0
        assert previous <= row['start_ticks'] < row['end_ticks']
        previous = row['end_ticks']
        assert math.isfinite(row['seconds']) and row['seconds'] > 0
        assert math.isclose(row['seconds'], (row['end_ticks']-row['start_ticks'])/row['frequency'], rel_tol=1e-14)
        error = check_result(row['result'], case['expected'], family=manifest['family'])
        assert math.isclose(row['maximum_centroid_error'], error, rel_tol=1e-12, abs_tol=1e-15)
        if row['name'] in first:
            assert row['result'] == first[row['name']]
        else:
            first[row['name']] = row['result']
