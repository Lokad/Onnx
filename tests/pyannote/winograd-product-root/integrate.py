"""Copy exactly the qualified three edits and four additions after application admission."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-winograd-product-root-integration-20260923'
SOURCE=ROOT/'artifacts/pyannote-winograd-product-source-v2-20260923'
APP=ROOT/'artifacts/pyannote-winograd-product-app-amd-20260923'
PRODUCT=ROOT/'artifacts/pyannote-winograd-product-amd-20260923'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert not BASE.exists()
    for folder in [APP,PRODUCT]:
        closed=read(folder/'closed.json');assert closed['passed']
        for name,wanted in closed['files'].items():assert pin(folder/name)==wanted,name
    assert pin(PRODUCT/'closed.json')['sha256']=='399dde04df28a204d83e4dcb27ed92c7f013af28343994f7e2f42539ec593b17'
    analysis=read(APP/'analysis.json')
    sys.path.insert(0,str(ROOT/'tests/pyannote/winograd-product-app-amd'))
    from admission import evaluate
    assert analysis['passed'] and evaluate(analysis['table'])==analysis['performance']
    assert analysis['performance']['admitted']
    assert (analysis['native_public_requests'],analysis['meeting_requests'],analysis['timing_requests'])==(24,3,96)
    assert analysis['identities']['candidate']==read(PRODUCT/'analysis.json')['measured']
    assert pin(SOURCE/'prepared.json')['sha256']=='3c7e4d2432558ef6044f6a484b64994c990762e192b4e77e93b74584e7dab8cd'
    prepared=read(SOURCE/'prepared.json')
    assert prepared['passed'] and len(prepared['before'])==416
    assert set(prepared['changed'])=={'src/Lokad.Onnx/GraphConvPacking.cs','src/Lokad.Onnx/TensorOps.ConvBlocked.cs','tests/Lokad.Onnx.Backend.Tests/ConvBlockedSpatialTests.cs'}
    assert set(prepared['added'])=={'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Winograd.cs','src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Winograd.Kernels.cs','src/Lokad.Onnx/Zzz.ConvWinogradDispatch.cs','tests/Lokad.Onnx.Backend.Tests/ConvWinogradTests.cs'}
    for name,wanted in prepared['files'].items():assert pin(SOURCE/name)==wanted,name
    for name,wanted in prepared['before'].items():assert pin(ROOT/name)==wanted,name
    for name in prepared['added']:assert not (ROOT/name).exists(),name
    qualified=read(PRODUCT/'bundle/stage.json')['files']
    names=[*prepared['before'],*prepared['added']];assert len(names)==len(set(names))==420
    for name in names:assert pin(SOURCE/'source'/name)==qualified['source/'+name],name
    BASE.mkdir()
    for name in prepared['changed']:
        backup=BASE/'original'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,backup)
    for name in [*prepared['changed'],*prepared['added']]:shutil.copy2(SOURCE/'source'/name,ROOT/name)
    after={name:pin(ROOT/name) for name in names}
    assert set(name for name,wanted in prepared['before'].items() if after[name]!=wanted)==set(prepared['changed'])
    assert all(after[name]==qualified['source/'+name] for name in after)
    receipt=dict(passed=True,application=pin(APP/'closed.json'),product=pin(PRODUCT/'closed.json'),
        prepared=pin(SOURCE/'prepared.json'),patch=pin(SOURCE/'candidate.patch'),changed=prepared['changed'],added=prepared['added'],
        source_files=after,originals={name:pin(BASE/'original'/name) for name in prepared['changed']},tool=pin(Path(__file__)),root_build_pending=True)
    (BASE/'applied.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,receipt=pin(BASE/'applied.json'),changed=prepared['changed'],added=prepared['added'],root_build_pending=True)))


if __name__=='__main__':main()
