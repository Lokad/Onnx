"""Recover inspection only; retain the completed build and failed first inventory."""
import ast
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-observed-dense-where-inventory-amd-20260924'
BUILD = ROOT/'artifacts/parakeet-observed-dense-where-build-amd-20260924'
MODELS = ROOT/'artifacts/parakeet-validated-composition-models-amd-20260924'


def previous_closed():
    prepared = read(BUILD/'prepared.json'); assert prepared['passed']
    for name, wanted in prepared['files'].items(): assert pin(ROOT/name) == wanted, name
    assert prepared['archive'] == pin(BUILD/'payload.tar.gz') and prepared['stage'] == pin(BUILD/'bundle/stage.json')
    receipt = read(BUILD/'collected/collection.json'); state = read(BUILD/'collected/identity.json')
    transfer = read(BUILD/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BUILD/'results.tar.gz') and transfer['receipt'] == pin(BUILD/'collected/collection.json')
    assert receipt['terminal'] and receipt['code'] == state['code'] == 1 and receipt['input_error'] is None and state['complete']
    assert [r['name'] for r in state['runs']] == ['sdk-version','cli-restore','cli-build','inventory']
    assert [r['code'] for r in state['runs']] == [0,0,0,-6]
    assert 'System.IO.FileNotFoundException' in (BUILD/'collected/logs/inventory.stderr').read_text()
    assert 'SixLabors.ImageSharp, Version=3.0.0.0' in (BUILD/'collected/logs/inventory.stderr').read_text()
    assert not (BUILD/'collected/inventory/instructions.json').exists()
    for name, wanted in receipt['files'].items(): assert pin(BUILD/'collected'/name) == wanted, name
    assert pin(MODELS/'closed.json')['sha256'] == 'a71906951cccb561c71b747a12f815d8145efb1d7bd7dabcb61689bdf9bb3802'
    proof = read(MODELS/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(MODELS/name) == wanted, name
    assert read(BUILD/'collected/built.json')['passed']
    measured = read(BUILD/'payload.json')['measured']
    assert {n:pin(MODELS/'collected/runtimes/candidate'/n) for n in measured} == measured


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','il_body.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    copy(TOOLS.parent/'observed-dense-where-build-amd/remote.py', bundle/'tools/retained_remote.py')
    copy(TOOLS/'README.md', bundle/'README.md')
    for name in ['payload.json','identity.json','built.json','collection.json']:
        copy(BUILD/'collected'/name, bundle/'evidence'/('original-'+name))
    copy(MODELS/'collected/collection.json', bundle/'evidence/reference-collection.json')
    copy(MODELS/'payload.json', bundle/'evidence/reference-payload.json')
    links = {}
    def link(folder, remote, prefix):
        for path in folder.iterdir():
            if path.is_file(): links[prefix+'/'+path.name] = dict(source=remote+'/'+path.name, identity=pin(path))
    parent = '/dev/shm/lokad-parakeet-observed-dense-where-build-20260924'
    reference = '/dev/shm/lokad-parakeet-validated-composition-models-20260924'
    link(BUILD/'collected/runtime', parent+'/runtime', 'runtime')
    link(BUILD/'collected/bridge', parent+'/bridge', 'bridge')
    link(MODELS/'collected/runtimes/candidate', reference+'/runtimes/candidate', 'measured')
    name = 'evidence/prior-composition.json'
    links[name] = dict(source=parent+'/'+name, identity=pin(BUILD/'bundle'/name))
    for name in ['checks.py','il_body.py']: assert pin(TOOLS/name) == pin(TOOLS.parent/'observed-dense-where-build-amd'/name)
    stage = dict(passed=True, parent=parent, reference=reference, links=links,
        measured=read(BUILD/'payload.json')['measured'], product=read(BUILD/'collected/built.json')['product'],
        original_collection=pin(BUILD/'collected/collection.json'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(__import__('json').dumps(dict(archive=pin(BASE/'payload.tar.gz'), jobs=['inventory'], unchanged_product=stage['product'])))


if __name__ == '__main__': prepare()
