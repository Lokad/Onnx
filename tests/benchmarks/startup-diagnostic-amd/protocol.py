"""Fixed baseline-only startup diagnostic; no performance score."""
import collections
import hashlib
import json
import math
from pathlib import Path

JOBS = ['sdk-version','producer-restore','producer-build','exporter-restore','exporter-build','tracer-version','a-capture','b-capture','a-export','b-export']
PROVIDERS = 'Microsoft-Windows-DotNETRuntime:0x1019:5,Lokad-Parakeet-MatMul-Diagnostic:0x1:4'

GIB = 1024**3
LIMITS = dict(preflight_available=12*GIB, build_preflight_available=12*GIB, preflight_tmpfs=3*GIB, rss=8*GIB,
              available=GIB, tmpfs=GIB, output=256*1024**2, artifacts=512*1024**2, seconds=900)


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
        assert m['expected_affinity'] in [[0],[2]] and m['affinity'] == m['expected_affinity'] and m['threads']
        assert all(t['affinity'] == m['expected_affinity'] for t in m['threads'])
