import importlib.util
from pathlib import Path
import json, shutil, traceback

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-convolution-portable-rows-20260921'
PRIOR=ROOT/'artifacts/pyannote-convolution-pool-v5-20260921'
QUALIFIED=ROOT/'artifacts/pyannote-convolution-pool-applications-v2-20260921'
KERNEL=ROOT/'artifacts/pyannote-portable-row-groups-v3-20260921'
FEED=ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('conv_portable_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor);monitor.BASE=BASE
pin,read,save,verify,terminal,psutil=monitor.pin,monitor.read,monitor.save,monitor.verify,monitor.terminal,monitor.psutil
def rel(path):return path.relative_to(ROOT).as_posix()
