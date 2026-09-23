"""Freeze a diagnostic capture of the already qualified standalone binary."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-winograd-range-codegen-amd-20260923'
NUM=ROOT/'artifacts/pyannote-winograd-range-numerics-amd-20260923'

def previous_closed():
    assert pin(NUM/'closed.json')['sha256']=='25cced10a2b855406144a0011625751263d8042290ce604c3d47f0818db95ae2'
    proof=read(NUM/'closed.json');assert proof['passed'] and proof['numerically_admitted']
    for name,wanted in proof['files'].items():assert pin(NUM/name)==wanted,name
    for name,wanted in read(NUM/'bundle/stage.json')['direct'].items():assert pin(ROOT/'src/Lokad.Onnx'/name)==wanted

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for p in (NUM/'collected/runtime').iterdir():
        if p.is_file():copy(p,bundle/'runtime'/p.name)
    copy(NUM/'collected/built.json',bundle/'built.json')
    copy(NUM/'bundle/source/global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    for name in ['closed.json','collected/collection.json','payload.json']:
        copy(NUM/name,bundle/'evidence'/Path(name).name)
    for width in [256,512]:copy(NUM/'collected'/f'captured-{width}'/'result.json',bundle/'evidence'/f'captured-{width}.json')
    copy(NUM/'bundle/evidence/fixtures.json',bundle/'evidence/fixtures.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,consumer=read(NUM/'collected/built.json')['consumer'],
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
