"""Keep the measured product; allow exactly the explicit-argument fixture repair."""
from pathlib import Path
import importlib.util
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-root-amd'
APP = TOOLS.parent/'attention-owned-pyannote-app-amd'
PREVIOUS = TOOLS.parent/'attention-owned-root-amd'
sys.path.insert(1, str(PARENT))
from protocol import pin, read
from fixture_repair import TEST, repair, fingerprint, verify_source_map


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


original = load('attention_original_source_scope', PREVIOUS/'source_scope.py')
SOURCE, BUILD, CENSUS, QUALIFIED = original.SOURCE, original.BUILD, original.CENSUS, original.QUALIFIED
TARGETS, CHANGED = original.TARGETS, original.CHANGED
ORIGINAL_FIXTURE = original.FIXTURE
FIXTURE = TOOLS/'OwnedAttentionPreparationTests.cs.txt'
FIXTURE_PIN = fingerprint(repair(ORIGINAL_FIXTURE.read_bytes()))
FAILED = ROOT/'artifacts/parakeet-attention-owned-root-amd-20260928'
FAILED_PIN = dict(bytes=122000, sha256='9c757ee055c982843f1a630b3ae06a0a57666c2fd85ef3a74a09bcb7d87ea9bc')
PREVIOUS_APPLIED = original.APPLIED
APPLIED = ROOT/'artifacts/parakeet-attention-owned-root-recovery-integration-20260928'
verify_root = original.verify_root


def verify_source():
    source = original.verify_source()
    assert pin(FAILED/'failed.json') == FAILED_PIN
    failure = read(FAILED/'failed.json')
    assert not failure['passed'] and not failure['release_admitted']
    assert failure['terminal'] and failure['code'] == 1
    for name, wanted in failure['files'].items(): assert pin(FAILED/name) == wanted, name
    assert pin(FIXTURE) == FIXTURE_PIN
    assert FIXTURE.read_bytes() == repair(ORIGINAL_FIXTURE.read_bytes())
    return source


def root_files(source):
    result = dict(source['source']); result[TEST] = FIXTURE_PIN
    verify_source_map(source['source'], result, FIXTURE.read_bytes())
    return result


if __name__ == '__main__':
    source = verify_source(); verify_root(root_files(source))
    print(dict(passed=True, source_files=446, product_source_unchanged=True,
               repair=TEST, failed_run=FAILED_PIN))
