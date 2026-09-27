"""Bind the unchanged measured LSTM product and two portable facts before integration."""
import importlib.util
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-root-amd'
APP = TOOLS.parent/'decoder-lstm-layout-pyannote-app-amd'
sys.path.insert(1, str(PARENT))
from protocol import pin, read

SOURCE = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-decoder-lstm-layout-scalar-baseline-20260927'
FIXTURE = ROOT/'artifacts/parakeet-decoder-lstm-layout-integration-tests-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
APPLIED = ROOT/'artifacts/parakeet-decoder-lstm-layout-root-integration-20260927'
TARGETS = ['src/Lokad.Onnx/GraphLstmPacking.cs', 'src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs']
HELPER = 'src/Lokad.Onnx/PreparedLstmProjection.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/PreparedLstmProjectionTests.cs'
CAPTURE = 'tests/Lokad.Onnx.Backend.Tests/LstmLayoutContracts.cs'
CHANGED = [*TARGETS, HELPER, TEST]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_source():
    assert pin(QUALIFIED/'closed.json')['sha256'] == 'd0a78cdd3106d6a72a41303f879bbb2f9ea3bd6a298d778333015f38ccdac246'
    proof = read(QUALIFIED/'closed.json'); assert proof['passed']
    applied = QUALIFIED/'bundle/evidence/root-applied.json'
    assert pin(applied) == proof['files']['bundle/evidence/root-applied.json']
    before = read(applied)['source_files']; assert len(before) == 441
    assert pin(SOURCE/'failed.json')['sha256'] == '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'
    failure = read(SOURCE/'failed.json')
    assert not failure['passed'] and failure['evidence_verified'] and failure['terminal']
    stage = SOURCE/'bundle/stage.json'
    assert pin(stage) == failure['files']['bundle/stage.json']
    measured = read(stage)['source']
    assert len(measured) == 443 and set(measured) == set(before) | {HELPER, CAPTURE}
    assert {n for n, wanted in measured.items() if before.get(n) != wanted} == {*TARGETS, HELPER, CAPTURE}
    for name, wanted in measured.items():
        assert pin(SOURCE/'bundle/source'/name) == wanted == failure['files']['bundle/source/'+name], name
    script = TOOLS.parent/'decoder-lstm-layout/source.py'
    assert pin(script) == failure['files']['frozen-tools/source.py']
    builder = load('fixed_lstm_layout_source', script)
    originals = {name:(QUALIFIED/'bundle/source'/name).read_text(encoding='utf8') for name in builder.SOURCE_FILES}
    changed = builder.changed(originals)
    for name, text in changed.items():
        assert text == (SOURCE/'bundle/source'/name).read_text(encoding='utf8'), name
    assert pin(CONTRACTS/'closed.json')['sha256'] == '63c97822999a74921c2e8a3c64e0af9ec6682a6829ba83de52bba353b93b85ec'
    diagnostic = read(CONTRACTS/'closed.json'); contracts = read(CONTRACTS/'analysis.json')
    assert diagnostic['passed'] and diagnostic['diagnostic_only'] and not diagnostic['original_campaign_passed']
    assert diagnostic['analysis'] == pin(CONTRACTS/'analysis.json')
    assert not contracts['contract_regression_found'] and contracts['compiled']['passed']
    assert contracts['original_failure'] == pin(SOURCE/'failed.json')
    assert contracts['baseline'] == read(QUALIFIED/'analysis.json')['built']
    assert contracts['candidate'] == read(SOURCE/'collected/built.json')['candidate']
    assert pin(FIXTURE/'prepared.json')['sha256'] == 'c091bf7b179a6a90fc8337bdb2885818ba7f04819257bb2f7bbf7738b16f5e6c'
    fixture = read(FIXTURE/'prepared.json')
    assert fixture['passed'] and fixture['product_unchanged'] and not fixture['root_applied']
    assert not fixture['compiled_or_executed'] and fixture['source_stage'] == pin(stage)
    adaptation = load('portable_lstm_layout_fixture', TOOLS.parent/'decoder-lstm-layout-integration/prepare_tests.py')
    assert fixture == dict(**adaptation.review(), patch=pin(FIXTURE/'review.patch'))
    assert fixture['fixture'] == pin(FIXTURE/'PreparedLstmProjectionTests.cs')
    policy = 'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    assert fixture['source_policy_test'] == before[policy] == measured[policy]
    source = {n:wanted for n,wanted in measured.items() if n != CAPTURE}
    return dict(source=source, before=before, source_stage=pin(stage), fixture=pin(FIXTURE/'prepared.json'))


def root_files(source):
    result = dict(source['source']); result[TEST] = pin(FIXTURE/'PreparedLstmProjectionTests.cs')
    assert len(result) == 443 and CAPTURE not in result
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
