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


def definition(manifest):
    """Keep the release-v1 schedule fixed; v2 declares its schedule up front."""
    if manifest['schema'] == 1:
        assert (manifest['warmup_passes'], manifest['measured_passes']) == (1, 3), 'audio-v1 pass definition changed'
        return 'audio-full-api-v1', dict(warmup_passes=1, measured_passes=3), 900
    assert manifest['schema'] == 2 and manifest['protocol'] == 'audio-sustained-v2', 'unknown audio protocol'
    conditioning, measured, timeout = (manifest[k] for k in ('conditioning_passes', 'measured_passes', 'process_timeout_seconds'))
    assert type(conditioning) is int and 0 <= conditioning <= 32, 'invalid conditioning count'
    assert type(measured) is int and 2 <= measured <= 32, 'invalid measured count'
    assert type(timeout) is int and 60 <= timeout <= 7200, 'invalid process timeout'
    assert 'warmup_passes' not in manifest, 'ambiguous audio-v2 schedule'
    return manifest['protocol'], dict(first_use_passes=1, conditioning_passes=conditioning, measured_passes=measured), timeout


def schedule(manifest, mode):
    assert mode in ('conformance', 'timing'), 'invalid audio mode'
    _, counts, _ = definition(manifest)
    if manifest['schema'] == 1:
        return ['warmup'] if mode == 'conformance' else ['warmup'] + ['measured'] * counts['measured_passes']
    return ['first-use'] if mode == 'conformance' else ['first-use'] + ['conditioning'] * counts['conditioning_passes'] + ['measured'] * counts['measured_passes']


def validate_records(value, manifest, mode):
    producer, counts, _ = definition(manifest)
    assert value['schema'] == manifest['schema'] and value['family'] == manifest['family'] and value['engine'] in ('managed', 'ort')
    assert value['conformance'] == (mode == 'conformance') and value['held_outputs_unchanged'] is True and value['affinity'] == 4
    assert math.isfinite(value['setup_seconds']) and value['setup_seconds'] >= 0
    assert not any(k.lower().startswith(('lokad_', 'dotnet_', 'complus_')) for k in value['flags'])
    if manifest['schema'] == 2:
        assert value['producer'] == producer and value['schedule'] == counts, 'audio protocol/schedule mismatch'
        startup = value['startup']
        assert all(type(startup[k]) is int for k in ('entry_ticks', 'setup_start_ticks', 'setup_end_ticks', 'frequency'))
        assert 0 <= startup['entry_ticks'] <= startup['setup_start_ticks'] <= startup['setup_end_ticks'] and startup['frequency'] > 0
        assert math.isclose(value['setup_seconds'], (startup['setup_end_ticks']-startup['setup_start_ticks'])/startup['frequency'], rel_tol=1e-12, abs_tol=1e-9)
        assert math.isclose(startup['preparation_seconds'], (startup['setup_start_ticks']-startup['entry_ticks'])/startup['frequency'], rel_tol=1e-12, abs_tol=1e-9)
    phases = schedule(manifest, mode)
    wanted = [(iteration, phase, case) for iteration, phase in enumerate(phases) for case in manifest['cases']]
    assert len(value['records']) == len(wanted)
    previous, first = value['startup']['setup_end_ticks'] if manifest['schema'] == 2 else 0, {}
    for row, (iteration, phase, case) in zip(value['records'], wanted):
        assert row['name'] == case['name'] and row['pass'] == iteration
        assert row['phase'] == phase
        assert row['ownership'] is True and row['input_sha256'] == case['raw_sha256']
        assert all(type(row[k]) is int for k in ['start_ticks', 'end_ticks', 'frequency']) and row['frequency'] > 0
        assert previous <= row['start_ticks'] < row['end_ticks']
        previous = row['end_ticks']
        assert math.isfinite(row['seconds']) and row['seconds'] > 0
        assert math.isclose(row['seconds'], (row['end_ticks']-row['start_ticks'])/row['frequency'], rel_tol=1e-14)
        if manifest['schema'] == 2:
            assert row['frequency'] == value['startup']['frequency']
            if value['engine'] == 'managed':
                assert type(row['allocated_bytes']) is int and row['allocated_bytes'] >= 0
                assert math.isfinite(row['gc_pause_seconds']) and row['gc_pause_seconds'] >= 0
                assert all(type(row[k]) is int and row[k] >= 0 for k in ('heap_bytes_before', 'heap_bytes_after'))
                assert len(row['gc_before']) == len(row['gc_after']) == 3
                assert all(type(a) is int and type(b) is int and 0 <= a <= b for a, b in zip(row['gc_before'], row['gc_after']))
        error = check_result(row['result'], case['expected'], family=manifest['family'])
        assert math.isclose(row['maximum_centroid_error'], error, rel_tol=1e-12, abs_tol=1e-15)
        if row['name'] in first:
            assert row['result'] == first[row['name']]
        else:
            first[row['name']] = row['result']
