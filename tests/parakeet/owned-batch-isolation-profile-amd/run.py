"""Rebind the existing profile transport; expose capture only, with no build."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
ORIGINAL = TOOLS.parent/'packed-final-row-profile-amd'
BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-profile-amd-20260925'
REMOTE = '/dev/shm/lokad-parakeet-owned-batch-isolation-profile-20260925'
APP = ROOT/'artifacts/parakeet-owned-batch-isolation-release-app-amd-20260925'
REMOTE_APP = '/dev/shm/lokad-parakeet-owned-batch-isolation-release-app-20260925'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load('retained_profile_transport', ORIGINAL/'run.py')
original.BASE = BASE
original.REMOTE = REMOTE
original.APP = APP
original.REMOTE_APP = REMOTE_APP
original.PRELUDE = original.transport.PRELUDE+f'\nbase=Path({REMOTE!r})\nsys.path.insert(0,str(base))\n'
pin, read, write, ssh = original.pin, original.read, original.write, original.ssh
prepared = original.prepared


def prepare():
    from prepare import prepare as create
    create()


def stage():
    # Apply inference headroom before creating a namespace, since no build runs.
    prepared()
    ssh(original.PRELUDE+'''
assert psutil.virtual_memory().available>=11*1024**3
assert psutil.disk_usage('/dev/shm').free>=2*1024**3
print(json.dumps(dict(passed=True)))
''')
    original.stage()


def launch():
    from prepare import diagnostic_gates
    diagnostic_gates()
    original.launch('capture')


def observe():
    original.observe('capture')


def collect():
    original.collect('capture')


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    globals()[sys.argv[1]]()
