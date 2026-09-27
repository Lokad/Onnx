"""Reuse the original native observer after the selected root is qualified."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
ORIGINAL = TOOLS.parent/'ort-diagnosis-amd'
BASE = ROOT/'artifacts/parakeet-attention-owned-ort-profile-amd-20260928'
REMOTE = '/dev/shm/lokad-attention-owned-ort-profile-20260928'
APP = ROOT/'artifacts/parakeet-attention-owned-app-amd-20260928'
REMOTE_APP = '/dev/shm/lokad-attention-owned-app-20260928'
QUALIFIED_ROOT = ROOT/'artifacts/parakeet-attention-owned-root-amd-20260928'
ROOT_DIGEST = None
APP_DIGEST = '600e9e67e9a18ba324c65a2220cb26de9021d9eb718f26614af90e677e236cf4'
CANDIDATE_CORE = 'ee5218dbab0a970b0f20f273fc2a839bb9b96e9d5438791529eb11c731d66859'
CANDIDATE_DATA = '1ba343fd8b00fd85bddb33c57aaebf4217431955c40bbe45576448467e59c99f'
sys.path.insert(1, str(ORIGINAL))

loader = importlib.util.spec_from_file_location('unchanged_native_transport', ORIGINAL/'run.py')
transport = importlib.util.module_from_spec(loader)
loader.loader.exec_module(transport)
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE = BASE, REMOTE
transport.APP, transport.REMOTE_APP = APP, REMOTE_APP
# Keep TOOLS bound to ORIGINAL: observer, worker and their source checks are exact.
pin, read, write = transport.pin, transport.read, transport.write


def qualification():
    assert ROOT_DIGEST is not None, 'Require the actual qualified root before staging or inference'
    assert pin(QUALIFIED_ROOT/'closed.json')['sha256'] == ROOT_DIGEST
    assert pin(APP/'closed.json')['sha256'] == APP_DIGEST
    inputs = {}
    for folder in [APP, QUALIFIED_ROOT]:
        proof = read(folder/'closed.json')
        assert proof['passed']
        for name, wanted in proof['files'].items():
            assert pin(folder/name) == wanted, name
        inputs[(folder/'closed.json').relative_to(ROOT).as_posix()] = pin(folder/'closed.json')
    assert read(APP/'closed.json')['admitted']
    root = read(QUALIFIED_ROOT/'analysis.json')
    candidate = read(APP/'analysis.json')['identities']['candidate']
    assert candidate['Lokad.Onnx.dll']['sha256'] == CANDIDATE_CORE
    assert candidate['Lokad.Onnx.Data.dll']['sha256'] == CANDIDATE_DATA
    assert root['passed'] and root['root_source_verified'] and root['measured'] == candidate
    assert root['inventory'] == dict(passed=True, core_methods=3288, data_methods=697,
        public_surface_equal=True, assembly_attributes_equal=True,
        method_bodies_equal=True, implementation_flags_equal=True)
    applied = read(QUALIFIED_ROOT/'bundle/evidence/root-applied.json')
    for name, wanted in applied['source_files'].items():
        assert pin(ROOT/name) == wanted, name
    for name, wanted in root['built'].items():
        assert pin(QUALIFIED_ROOT/'collected/runtime'/name) == wanted, name
    for folder in [TOOLS, ORIGINAL]:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    return dict(passed=True, root=pin(QUALIFIED_ROOT/'closed.json'),
        application=pin(APP/'closed.json'), measured=candidate, built=root['built'],
        inputs=inputs, observer_rebuilt=False, attribution_only=True)


def stage():
    context = qualification()
    transport.stage()
    write(BASE/'qualification.json', context)


def launch():
    assert read(BASE/'qualification.json') == qualification()
    transport.launch()


observe, collect = transport.observe, transport.collect


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['review', 'stage', 'launch', 'observe', 'collect']
    if sys.argv[1] == 'review':
        result = qualification()
        print(json.dumps({k:v for k,v in result.items() if k != 'inputs'}))
    else:
        globals()[sys.argv[1]]()
