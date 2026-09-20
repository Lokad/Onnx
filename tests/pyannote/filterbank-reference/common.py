"""Identities and fixed scope for the complete saved filterbank corpus."""
from pathlib import Path
import hashlib, json, os, sys

THREADS = {name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS')}
os.environ.update(THREADS)
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / 'artifacts/wespeaker-frontend-20260919'
PRODUCT = ROOT / 'src/Lokad.Onnx.Data/WeSpeakerAudio.cs'
STAGES = ('windowed', 'real', 'imaginary', 'power', 'energy', 'raw', 'features')
LIMITS = dict(seconds=180, rss=2 * 1024**3, available=1024**3, preflight=4 * 1024**3, disk=2 * 1024**3)
REFERENCE_LIMIT = 1e-8
ORIGINAL_LIMIT = 1e-4

def psutil_module():
    sys.path.append(str(ROOT / 'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    return psutil

def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())

def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)

def verify(files):
    for name, expected in files.items():
        assert pin(ROOT / name) == expected, name

def absent(identity):
    ps = psutil_module()
    try:
        return ps.Process(identity['pid']).create_time() != identity['birth']
    except ps.NoSuchProcess:
        return True

def metric(actual, expected, limit):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a - b
    scaled = np.abs(delta) / np.maximum(1, np.abs(b))
    index = int(scaled.argmax())
    return dict(values=a.size, max_scaled=float(scaled.flat[index]), max_absolute=float(np.abs(delta).max()),
                rms=float(np.sqrt(np.mean(delta * delta))), failed=int(np.count_nonzero(scaled > limit)),
                coordinate=[int(v) for v in np.unravel_index(index, a.shape)], actual=float(a.flat[index]), expected=float(b.flat[index]))

def libraries(process):
    files = {}
    for item in process.memory_maps():
        path = Path(item.path)
        if path.suffix.lower() in ('.dll', '.pyd') and any(part.lower() in ('numpy', 'numpy.libs', 'torch') for part in path.parts):
            files[str(path.resolve())] = pin(path)
    return files

def openblas_threads():
    import ctypes
    paths = list((Path(np.__file__).parent.parent / 'numpy.libs').glob('*openblas*.dll'))
    assert len(paths) == 1
    library = ctypes.CDLL(str(paths[0]))
    get = library.scipy_openblas_get_num_threads64_
    get.restype = ctypes.c_int
    return get()
