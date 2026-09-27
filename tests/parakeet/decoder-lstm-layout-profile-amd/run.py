"""Reuse the reviewed managed observer and original capture transport."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
ORIGINAL = TOOLS.parent/'packed-final-row-profile-amd'
PARENT = TOOLS.parent/'owned-batch-isolation-profile-amd'
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-profile-amd-20260927'
REMOTE = '/dev/shm/lokad-lstmlayout-profile-20260927'


def load(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


native = load('selected_native_profile_adapter', TOOLS.parent/'decoder-lstm-layout-ort-profile-amd/run.py')
APP, REMOTE_APP, QUALIFIED_ROOT = native.APP, native.REMOTE_APP, native.QUALIFIED_ROOT
original = load('unchanged_managed_profile_transport', ORIGINAL/'run.py')
original.BASE, original.REMOTE = BASE, REMOTE
original.APP, original.REMOTE_APP = APP, REMOTE_APP
original.PRELUDE = original.transport.PRELUDE + f'\nbase=Path({REMOTE!r})\nsys.path.insert(0,str(base))\n'
pin, read, write, ssh = original.pin, original.read, original.write, original.ssh
prepared = original.prepared


def prepare():
    from prepare import prepare as create
    create()


def stage():
    prepared()
    ssh(original.PRELUDE + '''
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
