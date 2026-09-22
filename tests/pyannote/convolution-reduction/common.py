"""Fixed selected product and local preparation/collection paths."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-convolution-reduction-20260922'
PRIOR = ROOT / 'artifacts/pyannote-integrated-profile-20260922'
AMD = ROOT / 'artifacts/pyannote-portable-amd-execution-20260922'
PAYLOAD = ROOT / 'artifacts/pyannote-portable-amd-payload-20260922/payload'
REMOTE = '/dev/shm/lokad-pyannote-convolution-reduction-20260922'
REMOTE_SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
DOTNET = '/home/vermorel/.dotnet/dotnet'
SSH = ['ssh', '-i', 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'vermorel@74.178.91.76']
CORE = 'e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838'
DATA = '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


monitor = module('convolution_reduction_local_monitor', MONITOR)
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
