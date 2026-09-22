"""Full-model qualification for the isolated direct-output convolution."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts/pyannote-direct-composition-acceptance-v2-20260922'
BASE = ROOT / 'artifacts/pyannote-direct-models-20260922'
CORE = '19b9007d174cf0af4784ed90ba02af556ce7148ce7400f9173722689d32231f8'
DATA = 'cb6f86b0f587105f808f72514f3af33055f52e9ccea66a41ea69b31557d3ac75'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


monitor = module('direct_model_monitor', MONITOR)
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
# Reuse the resource auditor without altering its process, memory or time bounds.
auditor = module('direct_model_resources', ROOT / 'tests/parakeet/portable-models/common.py')
resources = auditor.resources


def rel(path):
    return path.relative_to(ROOT).as_posix()


def candidate():
    proof = read(MODEL / 'closed.json')
    assert proof['passed']
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    summary = read(MODEL / 'analysis.json')
    assert summary['passed'] and summary['instruction_review']['compiled_kernel_bodies_equal']
    assert [r['cases'] for r in summary['caller']] == [400, 400]
    assert [(r['passed'], r['skipped']) for r in summary['suites']] == [(21, 0), (21, 0), (3311, 93), (343, 0)]
    assert pin(MODEL / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')['sha256'] == DATA
    return proof
