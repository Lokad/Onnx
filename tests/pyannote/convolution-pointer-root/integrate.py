"""Apply the exact qualified one-file source only after full application admission."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-pointer-root-integration-20260923'
SOURCE=ROOT/'artifacts/pyannote-convolution-pointer-unroll-20260923'
APP=ROOT/'artifacts/pyannote-convolution-pointer-app-amd-20260923'
PRODUCT=ROOT/'artifacts/pyannote-convolution-pointer-product-amd-v2-20260923'
NAME='src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert not BASE.exists()
    for folder in [APP,PRODUCT]:
        closed=read(folder/'closed.json');assert closed['passed']
        for name,wanted in closed['files'].items():assert pin(folder/name)==wanted,name
    assert pin(PRODUCT/'closed.json')['sha256']=='4727eadceb9fb9add889b82fffb4b63a33cc9afb407d974eb6955205f008580f'
    analysis=read(APP/'analysis.json')
    sys.path.insert(0,str(ROOT/'tests/pyannote/convolution-pointer-app-amd'))
    from admission import evaluate
    assert analysis['passed'] and evaluate(analysis['table'])==analysis['performance']
    assert analysis['performance']['admitted']
    assert (analysis['native_public_requests'],analysis['meeting_requests'],analysis['timing_requests'])==(24,3,96)
    assert analysis['identities']['candidate']==read(PRODUCT/'analysis.json')['measured']
    assert pin(SOURCE/'prepared.json')['sha256']=='863f9c85c35c6e8a3610eedcaa1b0014c6a6a6a069fc0ad814b589f1733d10b9'
    prepared=read(SOURCE/'prepared.json')
    assert prepared['passed'] and prepared['changed']==[NAME] and len(prepared['before'])==416
    for name,wanted in prepared['files'].items():assert pin(SOURCE/name)==wanted,name
    for name,wanted in prepared['before'].items():assert pin(ROOT/name)==wanted,name
    qualified=read(PRODUCT/'bundle/stage.json')['files']
    for name in prepared['before']:
        assert pin(SOURCE/'source'/name)==qualified['source/'+name],name
    BASE.mkdir()
    shutil.copy2(ROOT/NAME,BASE/'original-kernel.cs')
    shutil.copy2(SOURCE/'source'/NAME,ROOT/NAME)
    after={name:pin(ROOT/name) for name in prepared['before']}
    assert [name for name,wanted in prepared['before'].items() if after[name]!=wanted]==[NAME]
    assert all(after[name]==qualified['source/'+name] for name in after)
    receipt=dict(passed=True,application=pin(APP/'closed.json'),product=pin(PRODUCT/'closed.json'),
        prepared=pin(SOURCE/'prepared.json'),patch=pin(SOURCE/'candidate.patch'),changed=[NAME],
        source_files=after,original=pin(BASE/'original-kernel.cs'),tool=pin(Path(__file__)),root_build_pending=True)
    (BASE/'applied.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,receipt=pin(BASE/'applied.json'),changed=[NAME],root_build_pending=True)))


if __name__=='__main__':main()
