import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-reduction-accuracy-20260921'
ORIGINAL = ROOT / 'artifacts/parakeet-projection-20260921'
PRODUCT = ROOT / 'artifacts/parakeet-performance-profile-v2-20260921/bin'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
loader = importlib.util.spec_from_file_location('qualified_monitor', MONITOR)
monitor = importlib.util.module_from_spec(loader)
loader.loader.exec_module(monitor)
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
monitor.BASE = BASE
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROUTES = ['managed-native', 'native-native', 'managed-managed', 'native-managed']
BLOCKS = [128, 256, 512, 4096]


def rel(p):
    return p.relative_to(ROOT).as_posix()


def array(p, spec):
    assert pin(p) == {k: spec[k] for k in ('bytes', 'sha256')}
    value = np.fromfile(p, dtype=spec['dtype']).reshape(spec['shape'])
    assert np.isfinite(value).all()
    return value


def tensors(folder):
    result = read(folder / 'result.json')
    assert result['complete']
    return {v['name']: array(folder / v['file'], v) for v in result['outputs']}
