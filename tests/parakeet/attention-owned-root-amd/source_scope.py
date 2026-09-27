"""Bind the measured one-method policy and its already executed portable fixture."""
import importlib.util
from pathlib import Path
import sys

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
PARENT=TOOLS.parent/'pad-current-root-amd'
APP=TOOLS.parent/'attention-owned-pyannote-app-amd'
sys.path.insert(1,str(PARENT))
from protocol import pin,read

SOURCE=ROOT/'artifacts/parakeet-attention-owned-source-20260928'
BUILD=CONTRACTS=ROOT/'artifacts/parakeet-attention-owned-build-amd-20260928'
CENSUS=ROOT/'artifacts/parakeet-attention-owned-census-amd-20260928'
FIXTURE=TOOLS.parent/'attention-owned-source/OwnedAttentionPreparationTests.cs.txt'
QUALIFIED=ROOT/'artifacts/parakeet-pointwise-tail-root-amd-20260927'
APPLIED=ROOT/'artifacts/parakeet-attention-owned-root-integration-20260928'
TARGETS=['src/Lokad.Onnx/GraphOwnedPacking.cs']
TEST='tests/Lokad.Onnx.Backend.Tests/OwnedAttentionPreparationTests.cs'
CHANGED=[*TARGETS,TEST]
SOURCE_PIN=dict(bytes=156324,sha256='ee998786951478b241232c66cad94b01e019fcdb9ae644b5b9731e47c3f1ac39')
FIXTURE_PIN=dict(bytes=12772,sha256='93da19d1fa2eb0c419ecaf90cf2405f09612e4d17e3a4ed12227185345561113')
BUILD_PIN=dict(bytes=4013,sha256='2a1bf8e3c2defbf806dd4aef0f409d6675ff3ea816f79127fd928e09a2e1da64')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_source():
    assert pin(QUALIFIED/'closed.json')['sha256']=='fc11676361a50ef613f783fcb51e4ead488c9c2ee34a3edddcbb561f67037e47'
    proof=read(QUALIFIED/'closed.json');assert proof['passed']
    assert pin(QUALIFIED/'bundle/stage.json')==proof['files']['bundle/stage.json']
    before={n.removeprefix('source/'):w for n,w in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(before)==445 and TEST not in before
    assert pin(SOURCE/'prepared.json')==SOURCE_PIN
    stage=read(SOURCE/'prepared.json')
    assert stage['passed'] and not stage['release_admitted'] and not stage['root_product_changed']
    assert stage['source_before']==before and stage['baseline']==pin(QUALIFIED/'closed.json')
    measured=stage['source']
    assert len(measured)==446 and set(measured)==set(before)|{TEST}
    assert {n for n,w in measured.items() if before.get(n)!=w}==set(CHANGED)
    for name,wanted in before.items():
        assert pin(QUALIFIED/'bundle/source'/name)==wanted==proof['files']['bundle/source/'+name],name
    for name,wanted in measured.items():assert pin(SOURCE/'source'/name)==wanted,name
    script=TOOLS.parent/'attention-owned-source/prepare.py'
    assert pin(script)==stage['tools']['prepare.py']
    builder=load('fixed_attention_policy_source',script)
    changed,patch=builder.change((QUALIFIED/'bundle/source'/TARGETS[0]).read_bytes())
    assert changed==(SOURCE/'source'/TARGETS[0]).read_bytes()
    assert patch==(SOURCE/'candidate.patch').read_text(encoding='utf8')
    assert stage['source_reversible'] and stage['arithmetic_leaves_unchanged']
    assert stage['changed_product_files']==TARGETS and not stage['added_product_files']
    assert stage['changed_methods']==['PrepareOwnedMatMulWeights'] and not stage['added_methods']
    assert stage['added_tests']==[TEST]
    assert (stage['expected_added_attention_weights'],stage['expected_retained_maps'],stage['maximum_packed_bytes'])==(92,37,268435456)
    assert pin(BUILD/'build-review.json')==BUILD_PIN
    built=read(BUILD/'build-review.json')
    assert built['passed'] and built['source']==SOURCE_PIN and built['arithmetic_leaves_unchanged']
    assert built['product']['Lokad.Onnx.dll']['sha256']=='ee5218dbab0a970b0f20f273fc2a839bb9b96e9d5438791529eb11c731d66859'
    assert read(BUILD/'bundle/spec.json')['before_product']==read(QUALIFIED/'analysis.json')['built']
    assert pin(FIXTURE)==FIXTURE_PIN==measured[TEST]
    assert (SOURCE/'source'/TEST).read_bytes()==FIXTURE.read_bytes()
    policy='tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    assert before[policy]==measured[policy]
    return dict(source=measured,before=before,source_stage=SOURCE_PIN,fixture=FIXTURE_PIN)


def root_files(source):
    result=dict(source['source'])
    assert len(result)==446 and result[TEST]==FIXTURE_PIN
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
