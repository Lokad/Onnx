import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-reduction-model-20260921'
ORIGINAL = ROOT / 'artifacts/parakeet-performance-profile-v2-20260921/bin'
PREVIOUS = ROOT / 'artifacts/parakeet-reduction-accuracy-20260921'
NATIVE = ROOT / 'artifacts/parakeet-transcription-20260919/frozen'
REFERENCE = NATIVE / 'reference/manifest.json'
CORPUS = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/parakeet.json'
BASELINE = ROOT / 'artifacts/pyannote-optimized-parakeet-20260921/baseline.json'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('qualified_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
monitor.BASE = BASE


def rel(path):
    return path.relative_to(ROOT).as_posix()
