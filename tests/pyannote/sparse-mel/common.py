import importlib.util
import json
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-sparse-mel-20260921'
PRIOR = ROOT / 'artifacts/pyannote-convolution-portable-rows-20260921'
QUALIFIED = ROOT / 'artifacts/pyannote-convolution-portable-applications-20260921'
COMPARISON = ROOT / 'artifacts/pyannote-convolution-portable-comparison-20260921'
RELEASE = ROOT / 'artifacts/pyannote-vector-bias-comparison-finish-20260921'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('sparse_mel_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()
