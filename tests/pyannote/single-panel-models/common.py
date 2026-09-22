"""Full-model qualification for the isolated direct-output convolution."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts/pyannote-single-panel-composition-20260922'
BASE = ROOT / 'artifacts/pyannote-single-panel-models-20260922'
CORE = '1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309'
DATA = '4e602d9f6a35a51277d6deb9d75779d84cecf0a3a433d1b4eb70b0006462cca4'
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
    assert [(r['passed'], r['skipped']) for r in summary['suites']] == [(23, 0), (23, 0), (3313, 93), (343, 0)]
    assert pin(MODEL / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')['sha256'] == DATA
    return proof
