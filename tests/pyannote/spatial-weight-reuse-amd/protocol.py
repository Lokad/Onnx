"""Frozen resource and input contracts for eight actual-product numerical jobs."""
import collections
import hashlib
import json
import math
from pathlib import Path

JOBS = ['raw-256', 'raw-512', 'wide-256', 'wide-512', 'span-256', 'span-512', 'layers-256', 'layers-512']
GIB = 1024**3
LIMITS = dict(preflight_available=12*GIB, preflight_tmpfs=3*GIB, rss=8*GIB,
              available=GIB, tmpfs=GIB, output=GIB, artifacts=2*GIB, seconds=900)


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def verify(base):
    spec = read(base/'payload.json'); assert spec['limits'] == LIMITS and spec['jobs'] == JOBS
    for name, wanted in spec['files'].items():
        path = (base/name).resolve(); assert path.is_relative_to(base.resolve())
        assert pin(path) == wanted, name
    for name, wanted in spec['external'].items(): assert pin(name) == wanted, name
    return spec


def check_sample(row):
    assert 0 <= row['seconds'] < LIMITS['seconds']
    assert row['rss'] == sum(m['rss'] for m in row['members']) < LIMITS['rss']
    assert row['available'] >= LIMITS['available'] and row['tmpfs'] >= LIMITS['tmpfs']
    assert row['output'] <= LIMITS['output'] and row['artifacts'] <= LIMITS['artifacts']
    for m in row['members']:
        assert m['affinity'] == [2] and m['threads']
        assert all(t['affinity'] == [2] for t in m['threads'])
