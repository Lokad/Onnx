"""Bind exact measured pointwise source and the independent portable boundary facts."""
import importlib.util
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-root-amd'
APP = TOOLS.parent/'pointwise-tail-pyannote-app-amd'
sys.path.insert(1, str(PARENT))
from protocol import pin, read

SOURCE = ROOT/'artifacts/parakeet-pointwise-tail-source-20260927'
BUILD = ROOT/'artifacts/parakeet-pointwise-tail-contracts-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-pointwise-tail-arithmetic-contracts-amd-20260927'
FIXTURE = TOOLS.parent/'pointwise-tail-source/PackedColumnRemainderTests.cs.txt'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-lstm-layout-root-amd-20260927'
APPLIED = ROOT/'artifacts/parakeet-pointwise-tail-root-integration-20260927'
TARGETS = ['src/Lokad.Onnx/MathOps.cs']
HELPER = 'src/Lokad.Onnx/MathOps.PackedColumnTails.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/PackedColumnRemainderTests.cs'
CHANGED = [*TARGETS, HELPER, TEST]
FIXTURE_PIN = dict(bytes=4156, sha256='2b47d1c2bcb6f692abaf26f9d15cb1bb87c933a6c974615587d6b70e2a89a838')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_source():
    assert pin(QUALIFIED/'closed.json')['sha256'] == 'efb99eea455647c64bbe0811c1b7d46387add59049ac81e48a926b250e7c42da'
    proof = read(QUALIFIED/'closed.json'); assert proof['passed']
    assert pin(QUALIFIED/'bundle/stage.json') == proof['files']['bundle/stage.json']
    before = {name.removeprefix('source/'): wanted for name, wanted in read(QUALIFIED/'bundle/stage.json')['files'].items()
              if name.startswith('source/')}
    assert len(before) == 443
    assert pin(SOURCE/'prepared.json')['sha256'] == '9be6d0381e417b435b030b467cabd1d723398ffb9fac294ef8ff91c9b6a6e64c'
    stage = read(SOURCE/'prepared.json')
    assert stage['passed'] and not stage['release_admitted'] and not stage['root_product_changed']
    assert stage['source_before'] == before and stage['baseline'] == pin(QUALIFIED/'closed.json')
    measured = stage['source']
    assert len(measured) == 444 and set(measured) == set(before) | {HELPER}
    assert {name for name, wanted in measured.items() if before.get(name) != wanted} == {*TARGETS, HELPER}
    for name, wanted in before.items():
        assert pin(QUALIFIED/'bundle/source'/name) == wanted == proof['files']['bundle/source/'+name], name
    for name, wanted in measured.items(): assert pin(SOURCE/'source'/name) == wanted, name
    script = TOOLS.parent/'pointwise-tail-source/prepare.py'
    assert pin(script) == stage['tools']['prepare.py']
    builder = load('fixed_pointwise_source', script)
    changed, patch = builder.change((QUALIFIED/'bundle/source'/TARGETS[0]).read_bytes())
    assert changed == (SOURCE/'source'/TARGETS[0]).read_bytes()
    assert ''.join(patch) == (SOURCE/'candidate.patch').read_text(encoding='utf8')
    assert stage['full_panel_source_unchanged']
    assert stage['changed_methods'] == ['mm_unsafe_vectorized_intrinsics_2x4packed_bump']
    assert stage['added_methods'] == ['PackedColumnTailEightRows', 'PackedColumnMaskedEightRows']
    assert pin(BUILD/'build-review.json')['sha256'] == '51090680e6b172287122ef15c5f7e5a3ae2eaa41f083a76cdcb314f418ddc227'
    built = read(BUILD/'build-review.json')
    assert built['passed'] and built['source'] == pin(SOURCE/'prepared.json')
    assert built['products']['baseline'] == read(QUALIFIED/'analysis.json')['built']
    assert built['products']['candidate']['Lokad.Onnx.dll']['sha256'] == '7cac67880fa9a4d519ac18e5887f47f48f0f14903bdf74cc6561b45c851e4f27'
    assert pin(FIXTURE) == FIXTURE_PIN and TEST not in measured
    policy = 'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    assert before[policy] == measured[policy]
    return dict(source=measured, before=before, source_stage=pin(SOURCE/'prepared.json'), fixture=pin(FIXTURE))


def root_files(source):
    result = dict(source['source']); result[TEST] = pin(FIXTURE)
    assert len(result) == 445
    return result


def verify_root(files):
    for name, wanted in files.items(): assert pin(ROOT/name) == wanted, name
    prefixes = ['src/', 'tests/Lokad.Onnx.Backend.Tests/', 'tests/Lokad.Onnx.Tensors.Tests/']
    actual = {p.relative_to(ROOT).as_posix() for prefix in prefixes for p in (ROOT/prefix).rglob('*')
              if p.is_file() and not {'bin','obj'} & set(p.relative_to(ROOT).parts)}
    assert actual == {name for name in files if any(name.startswith(prefix) for prefix in prefixes)}
    return True


if __name__ == '__main__':
    source = verify_source(); verify_root(source['before'])
    print(dict(passed=True, source_files=len(root_files(source)), changed=CHANGED,
               root_applied=False, portable_fixture_compiled=False))
