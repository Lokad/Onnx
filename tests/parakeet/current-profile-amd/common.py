"""Fixed selected product and local preparation/collection paths."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-current-profile-amd-20260923'
PRIOR = ROOT / 'artifacts/pyannote-amd-profile-v3-20260922'
AMD = ROOT / 'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922'
PAYLOAD = ROOT / 'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload'
REMOTE = '/dev/shm/lokad-parakeet-current-profile-20260923'
REMOTE_SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
DOTNET = '/home/vermorel/.dotnet/dotnet'
SSH = ['ssh', '-i', 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'vermorel@74.178.91.76']
CORE = '521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb'
DATA = 'f3b9aa81ee9766797e95216714dec559c5d8b020f8df510cf5f7ee0dda82693a'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


monitor = module('amd_profile_local_monitor', MONITOR)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path): return path.relative_to(ROOT).as_posix()


def new_state():
    own = psutil.Process()
    return dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])


def prepared():
    from prepare import previous_closed
    spec = read(BASE/'prepared.json');assert spec['passed'];verify(spec['files'])
    assert pin(BASE/'payload.tar.gz')==spec['archive'] and pin(BASE/'payload/payload.json')==spec['payload']
    previous_closed()
    return spec
