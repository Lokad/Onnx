"""Freeze the scored consumer, original qualified component and all clocks/gates."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
from score import fixtures,constants
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-winograd-screen-amd-20260923'
NUM=ROOT/'artifacts/pyannote-winograd-numerics-amd-20260923'
CODE=ROOT/'artifacts/pyannote-winograd-codegen-amd-20260923'
REVIEW=ROOT/'artifacts/pyannote-winograd-codegen-review-20260923'

def previous_closed():
    for folder,digest in [(NUM,'8aae19686508fc7623263da795a259c8bffaa7c4802a53916638ca1bd781e319'),(CODE,'28c8d94684024251f515d5902757056ae59aff9824a35df5886a589024446566')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(NUM/'analysis.json')['numerically_admitted']
    assert pin(REVIEW/'review.json')['sha256']=='1d1b2d66a19b230c87c95c1b5bc549d6497539f4db3fd8ab9b06f024d3be932d'
    review=read(REVIEW/'review.json');assert review['passed'] and review['closure']==pin(CODE/'closed.json')
    for row in review['all_bodies']:assert pin(ROOT/'tests/pyannote/winograd-results'/row['file'])==row['digest']
    for name,wanted in read(NUM/'bundle/stage.json')['direct'].items():assert pin(ROOT/'src/Lokad.Onnx'/name)==wanted

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for p in (NUM/'collected/runtime').iterdir():
        if p.is_file():copy(p,bundle/'runtime'/p.name)
    for name in ['Screen.cs','Screen.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(NUM/'bundle/source/global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    for folder,label in [(NUM,'numerics'),(CODE,'codegen')]:copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    copy(CODE/'collected/collection.json',bundle/'evidence/collection.json')
    copy(CODE/'payload.json',bundle/'evidence/payload.json');copy(REVIEW/'review.json',bundle/'evidence/review.json')
    copy(NUM/'collected/captured-512/result.json',bundle/'reference.json')
    copy(NUM/'bundle/evidence/fixtures.json',bundle/'evidence/fixtures.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    calls=fixtures(read(bundle/'evidence/fixtures.json'))
    manifest=[]
    for c in calls:
        work=c['weights']['shape'][0]*c['input']['shape'][1]*9*c['output']['shape'][2]*c['output']['shape'][3]
        iterations=((1<<31)+work-1)//work;assert iterations==3
        manifest.append(dict(fixture=c['case'],index=c['index'],form=c['form'],work=work,iterations=iterations))
    assert len(constants(calls))==29
    save(bundle/'geometry.json',dict(calls=manifest,iterations_per_pass=261,clocks=4176,measured=3132,warmup=1044,preparation_clocks=464))
    save(bundle/'stage.json',dict(passed=True,component=pin(bundle/'runtime/WinogradPrototype.dll'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))

if __name__=='__main__':prepare()
