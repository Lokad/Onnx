"""Bounded AVX-512-first composition on the qualified normal source build."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-combined-avx512-20260922'
SOURCE = ROOT / 'artifacts/pyannote-portable-integration-tests-20260922'
PRIOR = ROOT / 'artifacts/pyannote-portable-applications-20260922'
BUILD = ROOT / 'artifacts/pyannote-portable-integration-20260922'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


monitor = module('combined_avx512_monitor', MONITOR)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
suite_tools = module('combined_avx512_suites', ROOT / 'tests/pyannote/portable-integration-tests/common.py')
suite_tools.BASE = BASE
read_suite = suite_tools.read_suite
RECEIPTS = [(SOURCE / 'closed.json', 'bf9822426cd75a86cc8c02c065005f0005e1a80b4ee63d5c53313fed145235f6'),
    (PRIOR / 'closed.json', '9d62d3d21a174e9be7105e4224b4b4b986b89dc6085ec60cc99b3aa2fdd7da77')]
SUITES = [('focused', 'FullyQualifiedName~Conv|FullyQualifiedName~PoolLifetime|FullyQualifiedName~GraphOwnership', False, 219, 2),
    ('hardware-disabled', 'FullyQualifiedName~ConvPortableRowsTests|FullyQualifiedName~ConvPackedRowsTests', True, 55, 2),
    ('backend-full', None, False, 3295, 95), ('tensors-full', None, False, 342, 0)]


def rel(path):
    return path.relative_to(ROOT).as_posix()


def audit_resources(state_path, expected):
    state = read(state_path)
    assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == expected
    identities, resources = [state['supervisor']], []
    for run in state['runs']:
        is_test = run['name'] in [s[0] for s in SUITES]
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
        assert run['preflight']['available'] >= (10 if is_test else 8) * 1024**3
        rows = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(r['rss'] for r in rows) == run['peak_rss']
        for row in rows:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3
            assert row['disk'] >= 20 * 1024**3 and row['output_bytes'] <= 1024**3
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        resources.append(dict(name=run['name'], seconds=run['seconds'], samples=len(rows), peak_rss=run['peak_rss']))
    for identity in identities:
        terminal(identity)
    return dict(identities=identities, resources=resources, resource_samples=sum(r['samples'] for r in resources),
        peak_rss=max(r['peak_rss'] for r in resources))
