"""Local bounded experiment; all qualified inputs and original monitors are read-only."""
import importlib.util
from pathlib import Path
import json, hashlib, math, shutil, sys, traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-complete-row-coverage-20260921'
RUNTIME = ROOT / 'artifacts/pyannote-convolution-pool-applications-v2-20260921/application-runtime'
SOURCE = ROOT / 'artifacts/pyannote-convolution-pool-v5-20260921/source/src/Lokad.Onnx'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('row_group_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
CORE = '0d098ba5fd3fd8799bb1dd018148123f296802c769db5f6148a1a5fa9d80118e'

def rel(path): return path.relative_to(ROOT).as_posix()
def state():
    p = psutil.Process()
    return dict(complete=False, code=None, supervisor=dict(pid=p.pid, birth=p.create_time()), runs=[])

