"""Reuse the existing single-process EventPipe limits and provider set."""
import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'dispatch-events-amd'
source = TOOLS/'protocol_base.py'
if not source.exists(): source = PARENT/'protocol.py'
loader = importlib.util.spec_from_file_location('pointwise_runtime_protocol', source)
inherited = importlib.util.module_from_spec(loader); loader.loader.exec_module(inherited)
JOBS = ['sdk-version', 'observer-restore', 'observer-build', 'tracer-version',
        'trace-capture', 'trace-export', 'trace-stacks']
GIB = 1024**3
LIMITS = dict(preflight_available=8*GIB, build_preflight_available=8*GIB,
    preflight_tmpfs=2*GIB, rss=4*GIB, available=GIB, tmpfs=GIB,
    output=64*1024**2, artifacts=256*1024**2, seconds=300)
PROVIDERS = inherited.PROVIDERS
inherited.JOBS, inherited.LIMITS = JOBS, LIMITS
read, pin, save, verify, check_sample = inherited.read, inherited.pin, inherited.save, inherited.verify, inherited.check_sample
