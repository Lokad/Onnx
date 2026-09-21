"""Fixed scope for the actual-input projection diagnostic; no product changes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import psutil

BASE = ROOT/'artifacts/parakeet-projection-20260921'
TRACE = ROOT/'artifacts/parakeet-layer-trace-v2-20260921'
DYNAMIC = ROOT/'artifacts/parakeet-layer-trace-dynamic-20260921'
REFERENCE = ROOT/'artifacts/parakeet-stem-reference-v3-20260921'
PROTOCOL = 'parakeet-layer-trace-v1'  # Frozen managed consumer's manifest format.
DIAGNOSTIC = 'parakeet-projection-own-input-v1'
STEM = '/pre_encode/out/Add_output_0'
RESHAPE = '/pre_encode/Reshape_output_0'
OUTPUTS = ['outputs', 'encoded_lengths', STEM, RESHAPE]
LIMITS = dict(seconds=900, rss=8*1024**3, available=1024**3,
              preflight_available=10*1024**3, disk=20*1024**3)
CAPTURES = [dict(id=engine+'-'+kind+suffix, engine=engine, input=kind, mode='trace')
            for kind, suffix in [('native', ''), ('managed', ''), ('native', '-repeat')]
            for engine in ('managed', 'native')]
JOBS = CAPTURES + [dict(id=engine, engine=engine) for engine in ('numpy', 'torch')]
ROUTES = [j['id'] for j in CAPTURES[:4]]
REFERENCE_LIMIT = 1e-9
ORIGINAL_LIMIT = 1e-4


def pin(path):
    path = Path(path)
    with path.open('rb') as f:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def rel(path):
    path = Path(path).resolve()
    try: return path.relative_to(ROOT).as_posix()
    except ValueError: return path.as_posix()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f: json.dump(value, f, indent=2, allow_nan=False)


def save(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def verify(spec):
    assert spec['protocol'] == PROTOCOL and spec['diagnostic'] == DIAGNOSTIC
    assert spec['jobs'] == JOBS and spec['limits'] == LIMITS and spec['outputs']['trace'] == OUTPUTS
    for name, expected in spec['files'].items(): assert pin(ROOT/name) == expected, name


def absent(identity):
    try: return psutil.Process(identity['pid']).create_time() != identity['birth']
    except psutil.NoSuchProcess: return True


def array(path, record):
    assert pin(path) == {k:record[k] for k in ('bytes', 'sha256')}
    result = np.fromfile(path, dtype=record['dtype']).reshape(record['shape'])
    assert np.isfinite(result).all()
    return result


def tensors(folder):
    result = read(folder/'result.json'); assert result['complete']
    return {v['name']:array(folder/v['file'], v) for v in result['outputs']}


def output(folder, name, value):
    assert np.isfinite(value).all()
    path = folder/(name+'.bin')
    with path.open('xb') as f: f.write(value.tobytes(order='C'))
    return dict(name=name, file=path.name, shape=list(value.shape), dtype=str(value.dtype), **pin(path))


def helpers():
    path = ROOT/'tests/pyannote/filterbank-reference/common.py'
    spec = importlib.util.spec_from_file_location('qualified_reference_helpers', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def coordinates():
    # No random-library lazy loads; cover the complete matrix deterministically.
    return sorted({0, 74*1024-1, *[int(i*(74*1024-1)//255) for i in range(256)]})


def metric(actual, expected, limit=ORIGINAL_LIMIT):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a-b; scaled = np.abs(delta)/np.maximum(1., np.abs(b)); index = int(scaled.argmax())
    return dict(values=a.size, max_scaled=float(scaled.flat[index]), failed=int(np.count_nonzero(scaled > limit)),
                rms=float(np.sqrt(np.mean(delta*delta))), max_absolute=float(np.abs(delta).max()),
                coordinate=[int(i) for i in np.unravel_index(index, a.shape)])
