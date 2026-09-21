"""Complete public qualification of the normal source/package runtime."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-portable-applications-20260922'
BUILD = ROOT / 'artifacts/pyannote-portable-integration-20260922'
COMPLETE = ROOT / 'artifacts/pyannote-portable-integration-completion-20260922'
TESTS = ROOT / 'artifacts/pyannote-portable-integration-tests-20260922'
PRIOR = ROOT / 'artifacts/pyannote-sparse-mel-applications-20260921'
MEETINGS = ROOT / 'artifacts/pyannote-optimized-meetings-20260921'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('portable_application_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
CORE = 'e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838'
DATA = '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
