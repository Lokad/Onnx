"""Fixed selected product and local preparation/collection paths."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-prepared-profile-amd-v2-20260922'
PRIOR = ROOT / 'artifacts/pyannote-amd-profile-v3-20260922'
AMD = ROOT / 'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922'
PAYLOAD = ROOT / 'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload'
REMOTE = '/dev/shm/lokad-pyannote-prepared-profile-v2-20260922'
REMOTE_SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
DOTNET = '/home/vermorel/.dotnet/dotnet'
SSH = ['ssh', '-i', 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'vermorel@74.178.91.76']
CORE = '3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
DATA = '6318cf48691470b908eec4c4d09c558172e43ce3b04bca9039c68966998a684b'
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
    spec = read(BASE / 'prepared.json'); assert spec['passed']; verify(spec['files'])
    assert pin(BASE / 'payload.tar.gz') == spec['archive']
    assert pin(BASE / 'payload/payload.json') == spec['payload']
    state = read(BASE / 'preparation.json'); assert state['complete'] and state['code'] == 0
    terminal(state['supervisor'])
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0
        for pid, birth in run['members'].items(): terminal(dict(pid=int(pid), birth=birth))
    return spec
