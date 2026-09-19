"""Recompute every saved pyannote backbone comparison independently of C#."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def child(root, name):
    path = (root / name).resolve()
    assert path.parent == root.resolve(), 'Fixture path differs'
    return path


def audit(manifest_path, result_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    result = json.loads(result_path.read_text(encoding='utf-8'))
    assert result['manifest_sha256'] == sha(manifest_path)
    assert manifest['scaled_absolute_tolerance'] == 1e-4
    assert not result['errors'] and result['rejections'] == 3
    names = ['synthetic-b1-f200', 'synthetic-b1-f201', 'synthetic-b1-f400', 'synthetic-b1-f800',
             'synthetic-b2-f200', 'english-16k', 'french-44k-stereo', 'jfk-48k-stereo']
    assert [c['name'] for c in manifest['cases']] == names
    assert [(r['repeat'], r['name']) for r in result['reports']] == [(repeat, name) for repeat in range(2) for name in names]
    native = {}
    assert len(manifest['files']) == 16
    for name, entry in manifest['files'].items():
        path = child(manifest_path.parent, name)
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        value = np.load(path, allow_pickle=False)
        assert str(value.dtype) == entry['dtype'] == 'float32' and list(value.shape) == entry['shape']
        assert np.isfinite(value).all()
        native[name] = value
    tensor_root = Path(str(result_path) + '.tensors')
    comparisons = values = failures = 0
    maximum = 0.0
    seen = set()
    first_hashes = {}
    for report in result['reports']:
        case = next(c for c in manifest['cases'] if c['name'] == report['name'])
        expected = native[case['output']]
        path = child(tensor_root, report['file'])
        assert report['file'] not in seen and sha(path) == report['sha256']
        seen.add(report['file'])
        assert path.stat().st_size == expected.size * 4 and report['shape'] == list(expected.shape)
        actual = np.fromfile(path, dtype='<f4').reshape(expected.shape)
        assert np.isfinite(actual).all()
        delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64)) / np.maximum(1, np.abs(expected.astype(np.float64)))
        error, index, bad = float(delta.max()), int(delta.argmax()), int((delta > 1e-4).sum())
        assert (error, index, bad) == (report['max_error'], report['worst_index'], report['bad'])
        assert report['passed'] == (bad == 0)
        if report['repeat'] == 0:
            first_hashes[report['name']] = report['sha256']
        else:
            assert first_hashes[report['name']] == report['sha256']
        values += actual.size
        failures += bad
        maximum = max(maximum, error)
        comparisons += 1
    assert {p.name for p in tensor_root.iterdir()} == seen
    assert values == result['values'] == 2764800 and comparisons == 16
    assert maximum == result['max_error'] and result['passed'] == (failures == 0)
    return dict(audit_consistent=True, passed=failures == 0, arrays=comparisons, values=values,
                maximum=maximum, failed_values=failures, manifest_sha256=sha(manifest_path), result_sha256=sha(result_path))


if __name__ == '__main__':
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
    raise SystemExit(0 if report['passed'] else 1)
