"""Qualification of the exact portable Pyannote / Parakeet arithmetic composition."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts/parakeet-single-panel-composition-20260922'
BASE = ROOT / 'artifacts/parakeet-single-panel-models-20260922'
CORE = 'abbf5e9878aeccf929a327ef4c1c3f5c9ff8afb8d9dec7ddb80455636cd1684d'
DATA = 'eb452663a09daa5d287fff1f65c07f2105f5c10ab474f845f210b2b30e3b921f'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
REFERENCE = ROOT / 'artifacts/parakeet-transcription-20260919/frozen/reference/manifest.json'
CORPUS = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/parakeet.json'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
SELECTED = ROOT / 'artifacts/pyannote-single-panel-models-20260922'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


monitor = module('portable_models_monitor', MONITOR)
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()


def candidate():
    assert pin(MODEL / 'closed.json')['sha256'] == '17e10dce9fdecf5aeb923ab9af0347c02a6850c6d5368bf4d8439eb76d01fc21'
    proof = read(MODEL / 'closed.json')
    assert proof['passed']
    verify(proof['files'])
    operands = read(MODEL / 'external-operand-proof.json')
    assert operands['passed']
    verify(operands['files'])
    for identity in proof['identities']:
        terminal(identity)
    assert pin(MODEL / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')['sha256'] == DATA
    return proof


def resources(base, state_name, jobs):
    state = read(base / state_name)
    assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == list(jobs)
    identities, observations = [state['supervisor']], []
    for run in state['runs']:
        minimum, rss, seconds, inference = jobs[run['name']]
        assert run['complete'] and run['code'] == 0 and run['preflight']['available'] >= minimum * 1024**3
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        rows = [json.loads(s) for s in (base / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(s['rss'] for s in rows) == run['peak_rss']
        assert rows[-1]['seconds'] <= run['seconds'] < seconds
        for row in rows:
            assert row['seconds'] < seconds and row['rss'] < rss * 1024**3
            assert row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3 and row['output_bytes'] <= 1024**3
            assert not inference or len(row['members']) <= 1
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        observations.append(dict(name=run['name'], samples=len(rows), peak_rss=run['peak_rss'], seconds=run['seconds']))
    for identity in identities:
        terminal(identity)
    return dict(identities=identities, resources=observations)


def close(base, analysis, files, identities):
    assert not (base / 'closed.json').exists() and not (base / 'analysis.json').exists()
    verify(files)
    save(base / 'analysis.json', analysis)
    files = dict(files)
    files.update({rel(p): pin(p) for p in base.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(base).parts)})
    save(base / 'closed.json', dict(passed=True, files=files, identities=identities))
    print(json.dumps(dict(closed=pin(base / 'closed.json'))), flush=True)
