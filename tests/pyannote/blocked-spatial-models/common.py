"""Full-model qualification of the normal prepared-convolution candidate."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
MODEL = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
BASE = ROOT/'artifacts/pyannote-blocked-spatial-models-20260922'
CORE = '3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
DATA = '6318cf48691470b908eec4c4d09c558172e43ce3b04bca9039c68966998a684b'
INPUT = ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


monitor = module('blocked_models_monitor', MONITOR)
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
auditor = module('blocked_model_resources', ROOT/'tests/parakeet/portable-models/common.py')
resources = auditor.resources


def rel(path): return path.relative_to(ROOT).as_posix()


def candidate():
    for folder, sha in [(MODEL, 'e7a9a30d88ef2a425c0c51b007e2b7d89428d54e50ef7dc4e446e191f95719e3'),
            (ROOT/'artifacts/pyannote-blocked-spatial-package-20260922', 'fc4f4811032ff38ea837b8d53ea68325a63cd9959b7e5d61b9fbb89ba7fefc66')]:
        assert pin(folder/'closed.json')['sha256'] == sha
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        for identity in proof['identities']: terminal(identity)
    summary = read(MODEL/'analysis.json')
    assert summary['passed'] and summary['product_source_identical']
    assert summary['instruction_review'] == dict(passed=True, unchanged_core=3161, unchanged_data=697, public_surface_equal=True)
    assert [(r['passed'], r['skipped']) for r in summary['suites']] == [(3344, 93), (343, 0), (31, 0)]
    assert pin(MODEL/'runtime/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(MODEL/'runtime/Lokad.Onnx.Data.dll')['sha256'] == DATA
    return read(MODEL/'closed.json')
