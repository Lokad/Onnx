"""Fixed selected product and local preparation/collection paths."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-selected-profile-amd-20260924'
PRIOR = ROOT / 'artifacts/pyannote-amd-profile-v3-20260922'
AMD = ROOT / 'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922'
PAYLOAD = ROOT / 'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload'
REMOTE = '/dev/shm/lokad-parakeet-selected-profile-20260924'
REMOTE_SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
DOTNET = '/home/vermorel/.dotnet/dotnet'
SSH = ['ssh', '-i', 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'vermorel@74.178.91.76']
CORE = '672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35'
DATA = '065b7a7f28561a37174f9d38ac13ef77bc59c010702e2b64200e9542376eb4c5'
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
