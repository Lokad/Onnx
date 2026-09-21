"""Shared identities and the unchanged bounded process monitor."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-request-comparison-20260921'
QUALIFIED = ROOT / 'artifacts/pyannote-request-contexts-v3-20260921'
PREDECESSOR = ROOT / 'artifacts/pyannote-optimized-ort-20260921'
OLD = ROOT / 'artifacts/audio-ort-baseline-v2-20260919'
INPUT = OLD / 'inputs/pyannote.json'
NATIVE = ROOT / 'tests/audio/comparison/native.py'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('bounded_process_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
monitor.BASE = BASE


def clean_env():
    prefixes = ('lokad_', 'dotnet_', 'complus_', 'omp_', 'mkl_', 'openblas_', 'blis_', 'numexpr_')
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith(prefixes)}
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in
        ['artifacts/pyannote-diarization-20260919/python', 'artifacts/pyannote-clustering-20260919/python']))
    env.update({k: '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']})
    return env


monitor.clean_env = clean_env


def verify_prepared(prepared):
    verify(prepared['files'])
    for name, wanted in prepared['external_files'].items():
        assert pin(Path(name)) == wanted, name


def public_auditor():
    spec = importlib.util.spec_from_file_location('original_audio_auditor', ROOT / 'tests/audio/comparison/audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_with_raw_hashes():
    import hashlib
    import numpy as np
    manifest = read(INPUT)
    assert (manifest['warmup_passes'], manifest['measured_passes'], len(manifest['cases'])) == (1, 3, 4)
    for case in manifest['cases']:
        pcm = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    return manifest
