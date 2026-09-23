"""Freeze diagnostic execution of both numerically qualified product DLLs."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-winograd-output-blocks-codegen-amd-v2-20260923'
FAILED=ROOT/'artifacts/pyannote-winograd-output-blocks-codegen-amd-20260923'
NUM=ROOT/'artifacts/pyannote-winograd-output-blocks-numerics-amd-20260923'
NUM_CLOSURE='47513c48326d17f7188a9dbf45bbd7232d4ee55122f82a3c4db23e8d12c8906b'


def previous_closed():
    assert pin(FAILED/'closed.json')['sha256']=='116194a103a73afba346d03d6325fe2bf7a011404b4ae905325b884b401512c9'
    failed=read(FAILED/'closed.json');assert not failed['passed'] and failed['retained_failure']
    for name,wanted in failed['files'].items():assert pin(FAILED/name)==wanted,name
    assert pin(NUM/'closed.json')['sha256']==NUM_CLOSURE
    proof=read(NUM/'closed.json');assert proof['passed'] and proof['numerically_admitted']
    for name,wanted in proof['files'].items():assert pin(NUM/name)==wanted,name
    source=read(ROOT/'artifacts/pyannote-winograd-output-blocks-source-20260923/prepared.json')
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed()
    assert pin(TOOLS/'numerical_checks.py')==pin(ROOT/'tests/pyannote/winograd-output-blocks-prototype/audit.py')
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for folder in ['runtime','runtimes/current','runtimes/candidate']:
        for p in (NUM/'collected'/folder).iterdir():
            if p.is_file():copy(p,bundle/folder/p.name)
    copy(NUM/'collected/built.json',bundle/'built.json')
    copy(NUM/'bundle/source/global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    for name in ['closed.json','collected/collection.json','payload.json']:
        copy(NUM/name,bundle/'evidence'/Path(name).name)
    for role in ['current','candidate']:
        for width in [256,512]:
            name=f'{role}-captured-{width}'
            copy(NUM/'collected'/name/'result.json',bundle/'evidence'/(name+'.json'))
    copy(NUM/'bundle/evidence/result.json',bundle/'evidence/fixtures.json')
    for name in ['closed.json','collected/collection.json','payload.json']:
        copy(FAILED/name,bundle/'evidence'/('failed-'+Path(name).name))
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,consumer=read(NUM/'collected/built.json')['consumer'],
        products=read(NUM/'payload.json')['products'],
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
