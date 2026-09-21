"""Frozen, local-only Parakeet encoder-boundary diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
sys.path.append(str(SITE))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
BASE = ROOT/'artifacts/parakeet-layer-trace-20260921'
OLD = ROOT/'artifacts/parakeet-transcription-20260919'
PRODUCT = ROOT/'artifacts/whisper-memory-product-v2-20260921/source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
CORE = 'd1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
PROTOCOL = 'parakeet-layer-trace-v1'
LIMITS = dict(seconds=900, rss=8*1024**3, available=1024**3,
              preflight_available=10*1024**3, disk=20*1024**3)


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def verify(spec):
    assert spec['protocol'] == PROTOCOL and spec['limits'] == LIMITS
    for name, expected in spec['files'].items():
        assert pin(ROOT/name) == expected, name


def absent(identity):
    import psutil
    try:
        return psutil.Process(identity['pid']).create_time() != identity['birth']
    except psutil.NoSuchProcess:
        return True


def array(path, record):
    import numpy as np
    assert pin(path) == {k:record[k] for k in ('bytes', 'sha256')}
    result = np.fromfile(path, dtype=record['dtype']).reshape(record['shape'])
    assert np.isfinite(result).all()
    return result
