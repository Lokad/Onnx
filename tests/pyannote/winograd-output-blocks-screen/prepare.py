"""Freeze identical complete Winograd call boundaries against two actual product DLLs."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
from score import fixtures,constants,PRODUCTS
from source_scope import generate
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-winograd-output-blocks-screen-amd-20260923'
NUM=ROOT/'artifacts/pyannote-winograd-output-blocks-numerics-amd-20260923'
CODE=ROOT/'artifacts/pyannote-winograd-output-blocks-codegen-amd-v2-20260923'
FAILED=ROOT/'artifacts/pyannote-winograd-output-blocks-codegen-amd-20260923'
REVIEW=ROOT/'artifacts/pyannote-winograd-output-blocks-codegen-review-20260923'


def previous_closed():
    for folder,digest,passed in [(NUM,'47513c48326d17f7188a9dbf45bbd7232d4ee55122f82a3c4db23e8d12c8906b',True),
        (CODE,'74a30abd4d5c61502c9f737991cf9bdd49a25addcd1c8ea17fbd4b4aea034b9d',True),
        (FAILED,'116194a103a73afba346d03d6325fe2bf7a011404b4ae905325b884b401512c9',False)]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']==passed
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(NUM/'analysis.json')['numerically_admitted']
    assert pin(REVIEW/'review.json')['sha256']=='bf99864117f3c77ac8cb3635c7db521a6c0fd783e0b6078c9a43407667ebf2e3'
    review=read(REVIEW/'review.json');assert review['passed'] and review['mechanism_admitted']
    assert review['first_closure']==pin(FAILED/'closed.json') and review['correction_closure']==pin(CODE/'closed.json')
    for row in review['raw_files']:
        assert pin(ROOT/row['source'])==row['pin']
        assert pin(ROOT/'tests/pyannote/winograd-output-blocks-results'/row['file'])==row['pin']
    source=read(ROOT/'artifacts/pyannote-winograd-output-blocks-source-20260923/prepared.json')
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed()
    assert (TOOLS/'Screen.cs').read_text(encoding='utf8')==generate()
    original=ROOT/'tests/pyannote/winograd-range-screen'
    for name in ['Screen.csproj','gates.py']:assert pin(TOOLS/name)==pin(original/name),name
    current=read(NUM/'collected/current-captured-512/result.json')
    candidate=read(NUM/'collected/candidate-captured-512/result.json')
    assert current['rows']==candidate['rows']
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for role in ['current','candidate']:
        for p in (NUM/'collected/runtimes'/role).iterdir():
            if p.is_file():copy(p,bundle/'runtimes'/role/p.name)
    for name in ['Screen.cs','Screen.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(original/'Screen.cs',bundle/'evidence/original-screen.cs')
    copy(NUM/'bundle/source/global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    for folder,label in [(NUM,'numerics'),(CODE,'codegen'),(FAILED,'failed-codegen')]:
        copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    copy(CODE/'collected/collection.json',bundle/'evidence/collection.json')
    copy(CODE/'payload.json',bundle/'evidence/payload.json');copy(REVIEW/'review.json',bundle/'evidence/review.json')
    copy(NUM/'collected/candidate-captured-512/result.json',bundle/'reference.json')
    copy(NUM/'bundle/evidence/result.json',bundle/'evidence/fixtures.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    calls=fixtures(read(bundle/'evidence/fixtures.json'));manifest=[]
    for c in calls:
        work=c['weights']['shape'][0]*c['input']['shape'][1]*9*c['output']['shape'][2]*c['output']['shape'][3]
        iterations=((1<<31)+work-1)//work;assert iterations==3
        manifest.append(dict(fixture=c['case'],index=c['index'],form=c['form'],work=work,iterations=iterations))
    assert len(constants(calls))==29
    save(bundle/'geometry.json',dict(calls=manifest,iterations_per_pass=261,clocks=4176,measured=3132,warmup=1044,preparation_clocks=464))
    products=read(NUM/'payload.json')['products']
    assert {role:files['Lokad.Onnx.dll']['sha256'] for role,files in products.items()}==PRODUCTS
    save(bundle/'stage.json',dict(passed=True,products=products,
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
