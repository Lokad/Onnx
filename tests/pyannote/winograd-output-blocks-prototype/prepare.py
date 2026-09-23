"""Freeze one unchanged numerical driver for both actual product DLLs."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save
from driver_scope import generate

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-winograd-output-blocks-numerics-amd-20260923'
BUILD = ROOT/'artifacts/pyannote-winograd-output-blocks-build-amd-20260923'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
PREVIOUS = ROOT/'artifacts/pyannote-winograd-range-numerics-amd-20260923'
SOURCE = ROOT/'artifacts/pyannote-winograd-output-blocks-source-20260923'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922/output'


def previous_closed():
    for folder,digest in [(BUILD,'cc8d46c59b8ee1328f99c781b9fdc4f2f3ae0ff0d1592fef752f888a250558dc'),
                          (CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'),
                          (PREVIOUS,'25cced10a2b855406144a0011625751263d8042290ce604c3d47f0818db95ae2')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name,wanted in proof['files'].items(): assert pin(folder/name) == wanted,name
    assert read(PREVIOUS/'closed.json')['numerically_admitted']
    source = read(SOURCE/'prepared.json')
    assert pin(SOURCE/'prepared.json')['sha256'] == '0a067addcd7425a0079ab1389d4c7d7aaa0e6b3e4a04f34ff1c85d686f0389e1'
    for name,wanted in source['before'].items(): assert pin(ROOT/name) == wanted,name
    for name,wanted in source['source'].items(): assert pin(SOURCE/'source'/name) == wanted,name
    assert read(BUILD/'analysis.json')['inventory']['unchanged_core_methods'] == 3178


def prepare():
    assert not BASE.exists(); previous_closed()
    assert (TOOLS/'Driver.cs').read_text(encoding='utf8') == generate()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['Driver.cs','KernelAccess.cs','Prototype.csproj']: copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    copy(ROOT/'tests/pyannote/winograd-range-prototype/Driver.cs',bundle/'evidence/original-driver.cs')
    for name in ['protocol.py','remote.py','remote_prepare.py']: copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m36-winograd-output-blocks-20260923.md',bundle/'prospective-plan.md')
    products = {}
    for role,folder in [('current',CURRENT/'collected/runtimes/current'),('candidate',BUILD/'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file(): copy(p,bundle/'runtimes'/role/p.name)
        products[role] = {name:pin(bundle/'runtimes'/role/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    assert products['current'] == read(BUILD/'analysis.json')['measured']
    assert products['candidate'] == read(BUILD/'analysis.json')['built']
    for name in ['closed.json','payload.json']: copy(BUILD/name,bundle/'evidence'/('build-'+name))
    copy(BUILD/'collected/collection.json',bundle/'evidence/build-collection.json')
    copy(PREVIOUS/'closed.json',bundle/'evidence/previous-closed.json')
    for name in ['raw-256','raw-512','captured-256','captured-512']:
        copy(PREVIOUS/'collected'/name/'result.json',bundle/'evidence'/('previous-'+name+'.json'))
    for name in ['result.json','capture-spec.json']: copy(FIXTURES/name,bundle/'evidence'/name)
    fixture_pins = {name:pin(FIXTURES/name) for name in ['result.json','capture-spec.json']}
    for entry in read(FIXTURES/'result.json')['tensors'].values():
        name = entry['path']; wanted = {k:entry[k] for k in ['bytes','sha256']}
        assert pin(FIXTURES/name) == wanted,name; fixture_pins[name] = wanted
    save(bundle/'stage.json',dict(passed=True,products=products,fixtures=fixture_pins,root_product_changed=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),products=products,fixtures=len(fixture_pins))))


if __name__ == '__main__': prepare()
