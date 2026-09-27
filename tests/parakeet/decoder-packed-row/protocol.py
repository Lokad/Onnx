"""Fixed build and correctness work for one prepared-row candidate."""
import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent / 'dispatch-events-amd'
source = TOOLS / 'protocol_base.py'
if not source.exists(): source = PARENT / 'protocol.py'
loader = importlib.util.spec_from_file_location('packed_row_base_protocol', source)
inherited = importlib.util.module_from_spec(loader); loader.loader.exec_module(inherited)
JOBS = ['inventory']
JOBS += [role + '-' + mode for mode in ['normal', 'noavx512', 'scalar'] for role in ['current', 'candidate']]
GIB = 1024**3
LIMITS = dict(preflight_available=2*GIB, build_preflight_available=2*GIB, preflight_tmpfs=GIB,
    rss=3*GIB, available=GIB, tmpfs=GIB, output=32*1024**2, artifacts=64*1024**2, seconds=300)
PROVIDERS = inherited.PROVIDERS
inherited.JOBS, inherited.LIMITS = JOBS, LIMITS
read, pin, save, verify, check_sample = inherited.read, inherited.pin, inherited.save, inherited.verify, inherited.check_sample
