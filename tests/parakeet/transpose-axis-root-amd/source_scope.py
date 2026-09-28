"""Bind the measured transpose dispatch and its already executed portable fixture."""
import importlib.util
from pathlib import Path
import sys

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
PARENT=TOOLS.parent/'pad-current-root-amd'
APP=TOOLS.parent/'transpose-axis-pyannote-app-amd'
sys.path.insert(1,str(PARENT))
from protocol import pin,read

SOURCE=ROOT/'artifacts/parakeet-transpose-axis-source-recovery-20260928'
BUILD=CONTRACTS=ROOT/'artifacts/parakeet-transpose-axis-build-recovery-amd-20260928'
FIXTURE=TOOLS.parent/'transpose-axis-source-recovery/TransposeAxisMovementTests.cs.txt'
QUALIFIED=ROOT/'artifacts/parakeet-attention-owned-root-recovery-amd-20260928'
APPLIED=ROOT/'artifacts/parakeet-transpose-axis-root-integration-20260928'
TARGETS=['src/Lokad.Onnx/TensorOps.Shape.cs']
TEST='tests/Lokad.Onnx.Backend.Tests/TransposeAxisMovementTests.cs'
CHANGED=[*TARGETS,TEST]
SOURCE_PIN=dict(bytes=156606,sha256='d311c9d133a89a1bcf341dfff64e6e6cf8cb75057f8c93bc94f10350aa79408e')
FIXTURE_PIN=dict(bytes=8615,sha256='425fcde8df7cd0f4ffaeadea31df42ea54c1806b0514bed91fb38055530095d2')
BUILD_PIN=dict(bytes=4378,sha256='f41cc8c3658a43c8348150f88df123691d6fdcee806a2844949aa3994c790d45')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_source():
    assert pin(QUALIFIED/'closed.json')['sha256']=='ee4a38ff671cc3fd8cd608c0dc3008f5c1b99f61ba6c139f3deeaf0e9039305e'
    proof=read(QUALIFIED/'closed.json');assert proof['passed']
    assert pin(QUALIFIED/'bundle/stage.json')==proof['files']['bundle/stage.json']
    before={n.removeprefix('source/'):w for n,w in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(before)==446 and TEST not in before
    assert pin(SOURCE/'prepared.json')==SOURCE_PIN
    stage=read(SOURCE/'prepared.json')
    assert stage['passed'] and not stage['release_admitted'] and not stage['root_product_changed']
    assert stage['source_before']==before and stage['baseline']==pin(QUALIFIED/'closed.json')
    measured=stage['source']
    assert len(measured)==447 and set(measured)==set(before)|{TEST}
    assert {n for n,w in measured.items() if before.get(n)!=w}==set(CHANGED)
    for name,wanted in before.items():
        assert pin(QUALIFIED/'bundle/source'/name)==wanted==proof['files']['bundle/source/'+name],name
    for name,wanted in measured.items():assert pin(SOURCE/'source'/name)==wanted,name
    script=TOOLS.parent/'transpose-axis-source-recovery/prepare.py'
    assert pin(script)==stage['tools']['prepare.py']
    builder=load('fixed_transpose_source',script)
    changed,patch=builder.change((QUALIFIED/'bundle/source'/TARGETS[0]).read_bytes())
    assert changed==(SOURCE/'source'/TARGETS[0]).read_bytes()
    assert patch==(SOURCE/'candidate.patch').read_text(encoding='utf8')
    assert stage['source_reversible'] and stage['arithmetic_leaves_unchanged']
    assert stage['changed_product_files']==TARGETS and not stage['added_product_files']
    assert stage['changed_methods']==['TransposeInto'] and not stage['added_methods']
    assert stage['added_tests']==[TEST] and stage['prediction_corpus_seconds_saved']==.45
    assert pin(BUILD/'build-review.json')==BUILD_PIN
    built=read(BUILD/'build-review.json')
    assert built['passed'] and built['source']==SOURCE_PIN and built['arithmetic_leaves_unchanged']
    assert built['product']['Lokad.Onnx.dll']['sha256']=='c471f5d1ead5889b00b141ce34f2a7689cfe179fbf513da4ce5db84ed01ce277'
    assert read(BUILD/'bundle/spec.json')['before_product']==read(QUALIFIED/'analysis.json')['built']
    assert pin(FIXTURE)==FIXTURE_PIN==measured[TEST]
    assert (SOURCE/'source'/TEST).read_bytes()==FIXTURE.read_bytes()
    policy='tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    assert before[policy]==measured[policy]
    return dict(source=measured,before=before,source_stage=SOURCE_PIN,fixture=FIXTURE_PIN)


def root_files(source):
    result=dict(source['source'])
    assert len(result)==447 and result[TEST]==FIXTURE_PIN
    return result


def verify_root(files):
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    prefixes=['src/','tests/Lokad.Onnx.Backend.Tests/','tests/Lokad.Onnx.Tensors.Tests/']
    actual={p.relative_to(ROOT).as_posix() for prefix in prefixes for p in (ROOT/prefix).rglob('*')
            if p.is_file() and not {'bin','obj'}&set(p.relative_to(ROOT).parts)}
    assert actual=={n for n in files if any(n.startswith(prefix) for prefix in prefixes)}
    return True


if __name__=='__main__':
    source=verify_source();verify_root(source['before'])
    print(dict(passed=True,source_files=len(root_files(source)),changed=CHANGED,
               root_applied=False,portable_fixture_previously_executed=True))
