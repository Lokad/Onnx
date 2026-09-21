"""Reuse the frozen process monitor for corrected-arithmetic application timing."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-arithmetic-comparison-20260921'
QUALIFICATION=ROOT/'artifacts/parakeet-reduction-dispatch-20260921'
OLD=ROOT/'artifacts/audio-ort-baseline-v2-20260919'
INPUT=OLD/'inputs/parakeet.json'
NATIVE=ROOT/'tests/audio/comparison/native.py'
PRODUCTION=ROOT/'artifacts/parakeet-performance-profile-v2-20260921/bin'
MONITOR_PATH=ROOT/'tests/parakeet/packing-budgets/common.py'
loader=importlib.util.spec_from_file_location('packing_budget_monitor',MONITOR_PATH)
monitor=importlib.util.module_from_spec(loader);loader.loader.exec_module(monitor)
pin,read,save,verify,terminal,psutil=monitor.pin,monitor.read,monitor.save,monitor.verify,monitor.terminal,monitor.psutil
monitor.BASE=BASE


def clean_env():
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_','omp_','mkl_','openblas_','blis_','numexpr_'))}
    env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in
        ('artifacts/pyannote-diarization-20260919/python','artifacts/pyannote-clustering-20260919/python')))
    env.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    return env


monitor.clean_env=clean_env
