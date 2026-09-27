"""Bind the measured two-path product and portable facts without applying root source."""
import importlib.util
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-root-amd'
APP = TOOLS.parent/'decoder-packed-row-pyannote-app-amd'
sys.path.insert(1, str(PARENT))
from protocol import pin, read

SOURCE = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v3-amd-20260927'
FIXTURE = ROOT/'artifacts/parakeet-decoder-packed-row-integration-tests-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
APPLIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-integration-20260927'
TARGET = 'src/Lokad.Onnx/TensorOps.MatMul.cs'
HELPER = 'src/Lokad.Onnx/PreparedSingleRowKernel.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/PreparedSingleRowTests.cs'
CHANGED = [TARGET, HELPER, TEST]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_source():
    assert pin(QUALIFIED/'closed.json')['sha256'] == 'f6df53ce2ab773898bc84f144abeda97908d383fef9a4df4b7299a22a4d3594d'
    proof = read(QUALIFIED/'closed.json'); assert proof['passed']
    applied = QUALIFIED/'bundle/evidence/root-applied.json'
    assert pin(applied) == proof['files']['bundle/evidence/root-applied.json']
    before = read(applied)['source_files']; assert len(before) == 439
    assert pin(SOURCE/'failed.json')['sha256'] == '6225233c00528cde3170a535ac973ddc7382c3af5086b370d90ff860d023d24d'
    original = read(SOURCE/'failed.json')
    assert original['evidence_verified'] and original['terminal'] and original['builds_passed']
    stage = SOURCE/'bundle/stage.json'
    assert pin(stage) == original['files']['bundle/stage.json']
    source = read(stage)['source']
    assert len(source) == 440 and set(source) == set(before) | {HELPER}
    assert {n for n, wanted in source.items() if before.get(n) != wanted} == {TARGET, HELPER}
    for name, wanted in source.items():
        assert pin(SOURCE/'bundle/source'/name) == wanted == original['files']['bundle/source/'+name], name
    script = TOOLS.parent/'decoder-packed-row/source.py'
    helper = TOOLS.parent/'decoder-packed-row/PreparedSingleRowKernel.cs.txt'
    for path in [script, helper]:
        assert pin(path) == original['files']['frozen-tools/'+path.name]
    builder = load('fixed_prepared_row_source', script)
    assert builder.changed((QUALIFIED/'bundle/source'/TARGET).read_text()) == (SOURCE/'bundle/source'/TARGET).read_text()
    assert source[HELPER] == pin(helper)
    assert pin(CONTRACTS/'closed.json')['sha256'] == 'fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d'
    contracts = read(CONTRACTS/'analysis.json')
    assert read(CONTRACTS/'closed.json')['analysis'] == pin(CONTRACTS/'analysis.json')
    assert contracts['passed'] and contracts['compiled']['passed']
    assert contracts['products']['current'] == read(QUALIFIED/'analysis.json')['built']['Lokad.Onnx.dll']
    assert contracts['products']['candidate'] == read(SOURCE/'collected/built.json')['products']['candidate']
    assert pin(FIXTURE/'prepared.json')['sha256'] == '8b3a18b8ee3e22d87f30c030c4c4558823235b0178f2d42d8fadc17f29c7a035'
    fixture = read(FIXTURE/'prepared.json')
    assert fixture['passed'] and fixture['product_unchanged'] and not fixture['root_applied']
    assert not fixture['compiled_or_executed'] and fixture['source_stage'] == pin(stage)
    adaptation = load('portable_prepared_row_fixture', TOOLS.parent/'decoder-packed-row-integration/prepare_tests.py')
    review = adaptation.review()
    assert fixture == dict(**review, patch=pin(FIXTURE/'review.patch'))
    assert fixture['fixture'] == pin(FIXTURE/'PreparedSingleRowTests.cs')
    policy = 'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    assert fixture['source_policy_test'] == before[policy] == source[policy]
    return dict(source=source, before=before, source_stage=pin(stage), fixture=pin(FIXTURE/'prepared.json'))


def root_files(source):
    result = dict(source['source']); result[TEST] = pin(FIXTURE/'PreparedSingleRowTests.cs')
    assert len(result) == 441
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
