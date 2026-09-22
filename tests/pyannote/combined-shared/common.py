"""Successor qualification utilities; the graph candidate and old tools stay frozen."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts/pyannote-combined-avx512-20260922'
CORE = 'e36963d848ff907acc521908c3a82569dd096c9e0e9ccad864a0d26028082d80'
DATA = '2b512f2527d3ede6e9a50b2206aabcb6bd4595c96c0ebcecdebe9e7a21939393'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('convolution_qualification_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()


def candidate():
    assert pin(MODEL / 'closed.json')['sha256'] == 'dd4cc7cc3bbde61e1de4b113657b72a27850bfa53dcd26c4dbeed2ef32f96e36'
    proof = read(MODEL / 'closed.json')
    assert proof['passed'] and read(MODEL / 'analysis.json')['passed']
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    assert pin(MODEL / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')['sha256'] == DATA
    return proof


def resources(base, state_name, jobs):
    """jobs maps each exact name to (preflight GiB, wall seconds, inference)."""
    state = read(base / state_name)
    assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == list(jobs)
    terminal(state['supervisor'])
    identities, observations = [state['supervisor']], []
    for run in state['runs']:
        minimum, seconds, inference = jobs[run['name']]
        assert run['complete'] and run['code'] in run['expected'] and run['preflight']['available'] >= minimum * 1024**3
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            identities.append(identity)
        rows = [json.loads(s) for s in (base / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(s['rss'] for s in rows) == run['peak_rss']
        assert rows[-1]['seconds'] <= run['seconds'] < seconds
        for row in rows:
            assert row['seconds'] < seconds and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3
            assert row['output_bytes'] <= 1024**3 and (not inference or len(row['members']) <= 1)
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        observations.append(dict(name=run['name'], samples=len(rows), peak_rss=run['peak_rss'], seconds=run['seconds']))
    return dict(identities=identities, resources=observations, samples=sum(r['samples'] for r in observations))


def close(base, analysis, files, identities):
    assert not (base / 'analysis.json').exists() and not (base / 'closed.json').exists()
    verify(files)
    save(base / 'analysis.json', analysis)
    files = dict(files)
    for path in base.rglob('*'):
        if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(base).parts):
            files[rel(path)] = pin(path)
    save(base / 'closed.json', dict(passed=True, files=files, identities=identities, analysis=pin(base / 'analysis.json')))
    print(json.dumps(dict(closed=pin(base / 'closed.json'))), flush=True)
