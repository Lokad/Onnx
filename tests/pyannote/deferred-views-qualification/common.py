"""Preserve the original cross-model gates for the one-method candidate."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts/pyannote-deferred-views-20260922'
CORE = '4f22824a7c315334982907f8846dc7e67dd0fddb1aadba4d908c89684285bbd9'
DATA = '92194232cd60979548cfe480db07a6f7a2e07d9f21dc8c69e133919c31ab4f2d'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


legacy = module('deferred_views_cross_model', ROOT / 'tests/pyannote/convolution-portable-qualification/common.py')
monitor = legacy.monitor
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
resources, close = legacy.resources, legacy.close


def rel(path):
    return path.relative_to(ROOT).as_posix()


def candidate():
    assert pin(MODEL / 'focused-closed.json')['sha256'] == '3c33d3fa6e9d2cde8877b500c7c2a68d73386040b1a1bc44b8636fea0828afd4'
    proof = read(MODEL / 'closed.json')
    assert proof['passed'] and read(MODEL / 'analysis.json')['captured_model_qualification_admitted']
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    assert pin(MODEL / 'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')['sha256'] == DATA
    return proof
