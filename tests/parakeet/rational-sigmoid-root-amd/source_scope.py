"""Bind the fixed sigmoid source and portable fixture to the actual qualified root."""
import importlib.util
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-root-amd'
APP = TOOLS.parent/'rational-sigmoid-pyannote-app-amd'
sys.path.insert(1, str(PARENT))
from protocol import pin, read

SOURCE = ROOT/'artifacts/parakeet-rational-sigmoid-source-20260927'
FIXTURE = ROOT/'artifacts/parakeet-rational-sigmoid-integration-tests-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-pad-current-root-amd-20260926'
APPLIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-integration-20260927'
TARGET = 'src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs'
HELPER = 'src/Lokad.Onnx/Zzz.SigmoidRational.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/SigmoidVectorTests.cs'
CHANGED = [TARGET, HELPER, TEST]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_source():
    assert pin(SOURCE/'prepared.json')['sha256'] == '1ca5891c2223e3fd9fa5c65c12c45e3bbeb96f45fc5903d11ccd950895577eb7'
    source = read(SOURCE/'prepared.json')
    assert source['passed'] and source['baseline'] == pin(QUALIFIED/'closed.json')
    assert source['baseline']['sha256'] == '71c80efd687562cba2e2b5d03e9e036b93de09d0a30f74b976b86355ed12fdf0'
    proof = read(QUALIFIED/'closed.json'); assert proof['passed']
    assert source['baseline_stage'] == proof['files']['bundle/stage.json'] == pin(QUALIFIED/'bundle/stage.json')
    before = {n.removeprefix('source/'): v for n, v in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(before) == 437 and len(source['source']) == 439
    assert set(source['source']) == set(before) | {HELPER, TEST}
    assert {n for n, v in source['source'].items() if before.get(n) != v} == set(CHANGED)
    assert source['changed_product_files'] == [TARGET] and source['added_product_files'] == [HELPER]
    assert source['added_tests'] == [TEST]
    for name, wanted in source['source'].items(): assert pin(SOURCE/'source'/name) == wanted, name
    for name, wanted in source['tools'].items(): assert pin(TOOLS.parent/'rational-sigmoid-source'/name) == wanted
    builder = load('original_rational_source', TOOLS.parent/'rational-sigmoid-source/prepare.py')
    assert builder.changed((QUALIFIED/'bundle/source'/TARGET).read_text()) == (SOURCE/'source'/TARGET).read_text()
    assert pin(FIXTURE/'prepared.json')['sha256'] == '96cf863fad425fc58c9a3dec3ba152184ee7b7b2ab66ac59d8f8e6a4b2dc70df'
    fixture = read(FIXTURE/'prepared.json')
    assert fixture['passed'] and fixture['product_unchanged'] and not fixture['root_applied']
    assert fixture['source_prepared'] == pin(SOURCE/'prepared.json')
    assert fixture['original'] == source['source'][TEST]
    assert fixture['eight_arithmetic_fact_bodies_unchanged'] and fixture['explicit_forwarding_overloads'] == 2
    script = TOOLS.parent/'rational-sigmoid-integration/prepare_tests.py'
    assert fixture['script'] == pin(script)
    adaptation = load('portable_sigmoid_fixture', script)
    expected, removed = adaptation.corrected_fixture((SOURCE/'source'/TEST).read_text())
    assert (FIXTURE/'SigmoidVectorTests.cs').read_text() == expected
    assert (FIXTURE/'runtime-guard-retained.txt').read_text() == removed
    assert fixture['facts'] == adaptation.FACTS
    assert fixture['corrected'] == pin(FIXTURE/'SigmoidVectorTests.cs')
    assert fixture['patch'] == pin(FIXTURE/'review.patch')
    assert fixture['qualification_guard_retained'] == pin(FIXTURE/'runtime-guard-retained.txt')
    guard = 'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    assert fixture['source_policy_test'] == before[guard] == source['source'][guard]
    return dict(source, before=before)


def root_files(source):
    result = dict(source['source']); result[TEST] = pin(FIXTURE/'SigmoidVectorTests.cs')
    assert len(result) == 439
    return result


def verify_root(files):
    for name, wanted in files.items(): assert pin(ROOT/name) == wanted, name
    prefixes = ['src/', 'tests/Lokad.Onnx.Backend.Tests/', 'tests/Lokad.Onnx.Tensors.Tests/']
    actual = {p.relative_to(ROOT).as_posix() for prefix in prefixes for p in (ROOT/prefix).rglob('*')
              if p.is_file() and not {'bin','obj'} & set(p.relative_to(ROOT).parts)}
    assert actual == {n for n in files if any(n.startswith(prefix) for prefix in prefixes)}
    return True


if __name__ == '__main__':
    source = verify_source(); verify_root(source['before'])
    print(dict(passed=True, source_files=len(root_files(source)), changed=CHANGED, root_applied=False))
