"""Independently recompute every saved Parakeet output comparison with NumPy."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def below(root, name):
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()) and path != root.resolve(), 'Path escapes root'
    return path


def audit(manifest_path, result_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result = json.loads(result_path.read_text(encoding='utf-8'))
    assert manifest['scope'] == 'parakeet-transcription' and manifest['schema'] == 1
    assert manifest['scaled_absolute_tolerance'] == 1e-4
    assert result['manifest_sha256'] == digest(manifest_path)
    assert result['application_passed'] and not result['errors'] and result['rejections'] == 6
    coverage = ['english-16k', 'french-44k-stereo', 'jfk-48k-stereo', 'english-token-limit',
                'english-frame-limit', 'silence', 'english-repeat']
    assert [c['name'] for c in manifest['cases']] == coverage
    assert [c['name'] for c in result['rows']] == coverage
    tensors = {}
    for name, entry in manifest['files'].items():
        path = below(manifest_path.parent, name)
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
        value = np.load(path, allow_pickle=False)
        assert list(value.shape) == entry['shape'] and str(value.dtype) == entry['dtype']
        assert np.isfinite(value).all()
        tensors[name] = value
    root = Path(str(result_path) + '.tensors')
    comparisons = values = 0
    maximum = 0.0
    failures = []
    seen = set()
    for case, row in zip(manifest['cases'], result['rows'], strict=True):
        assert row['actual'] == case['expected'], case['name']
        wanted = {(stage['model'], name): file for stage in case['stages'] for name, file in stage['outputs'].items()}
        wanted.update({('step-' + str(i), name): file for i, step in enumerate(case['steps'])
                       for name, file in step['outputs'].items()})
        assert len(row['comparisons']) == len(wanted)
        found = set()
        for entry in row['comparisons']:
            key = (entry['label'], entry['output'])
            assert key not in found and key in wanted
            found.add(key)
            expected = tensors[wanted[key]]
            path = below(root, entry['file'])
            assert entry['file'] not in seen and digest(path) == entry['sha256']
            seen.add(entry['file'])
            dtype = {'Float': np.dtype('<f4'), 'Int32': np.dtype('<i4'), 'Int64': np.dtype('<i8')}[entry['dtype']]
            assert dtype == expected.dtype and entry['shape'] == list(expected.shape)
            assert path.stat().st_size == expected.size * dtype.itemsize
            actual = np.fromfile(path, dtype=dtype).reshape(expected.shape)
            assert np.isfinite(actual).all()
            if dtype.kind == 'f':
                delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64)) / np.maximum(1, np.abs(expected.astype(np.float64)))
                error = float(delta.max()) if delta.size else 0.0
                index = int(delta.argmax()) if error else 0
            else:
                assert np.array_equal(actual, expected), key
                error = 0.0
                index = 0
            assert math.isclose(error, entry['max_error'], rel_tol=1e-14, abs_tol=1e-16)
            assert index == entry['worst_index'] and entry['passed'] == (error <= 1e-4)
            if error > 1e-4:
                failures.append(dict(case=case['name'], label=key[0], output=key[1], maximum=error, worst_index=index))
            maximum = max(maximum, error)
            comparisons += 1
            values += expected.size
        assert found == set(wanted)
    assert {p.name for p in root.iterdir()} == seen, 'Omitted or extra managed output'
    assert comparisons == result['comparisons'] == 784
    assert values == result['values_compared'] == 3090494
    assert maximum == result['max_error'] and result['passed'] == (not failures)
    return dict(audit_consistent=True, application_passed=True, numeric_gate_passed=not failures,
                manifest_sha256=digest(manifest_path), result_sha256=digest(result_path),
                arrays=comparisons, values=values, maximum=maximum, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = audit(args.manifest, args.result)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
    return 0 if report['numeric_gate_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
