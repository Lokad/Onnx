"""Fixed prepared-path comparison and unchanged fallback control, no variant sweep."""
import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'prepared-recurrence-timing-amd'
source = TOOLS/'protocol_base.py'
if not source.exists(): source = PARENT/'protocol.py'
loader = importlib.util.spec_from_file_location('layout_timing_protocol', source)
inherited = importlib.util.module_from_spec(loader); loader.loader.exec_module(inherited)
PREPARED_JOBS = [f'{role}-{duplicate}-{mode}' for mode in ['512', '256']
    for role, duplicate in [('selected', 0), ('candidate', 0), ('candidate', 1), ('selected', 1)]]
TIMING_JOBS = PREPARED_JOBS + [n.replace('selected', 'selectedfallback').replace('candidate', 'candidatefallback') for n in PREPARED_JOBS]
JOBS = ['sdk-version', 'timing-restore', 'timing-build', *TIMING_JOBS]
inherited.JOBS, inherited.TIMING_JOBS = JOBS, TIMING_JOBS
LIMITS = inherited.LIMITS
read, pin, save, verify, check_sample = inherited.read, inherited.pin, inherited.save, inherited.verify, inherited.check_sample
