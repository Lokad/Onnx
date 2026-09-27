"""Freeze exact full-model checks for the unchanged prepared-row candidate."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from compatibility import ROOT, CURRENT, QUALIFIED, FIRST, INVENTORY, CONTRACTS, SCREEN, DIAGNOSIS, review

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'observed-dense-where-models-amd'
PAD = TOOLS.parent/'pad-current-models-amd'
PREVIOUS = CURRENT
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-models-amd-20260927'
REMOTE_CURRENT = '/dev/shm/lokad-parakeet-rational-sigmoid-models-20260927'
REMOTE_QUALIFIED = '/dev/shm/lokad-parakeet-rational-sigmoid-root-20260927'
REMOTE_FIRST = '/dev/shm/lokad-decrow-20260927'
LABELS = dict(selected='Qualified current root', candidate='Prepared decoder single-row weights')
UNCHANGED = ['protocol.py', 'remote.py', 'checks.py', 'native_audit.py', 'public_audit.py']


def previous_closed():
    compatible = review()
    for name in UNCHANGED: assert pin(TOOLS/name) == pin(PAD/name), name
    expected = (PAD/'audit.py').read_text().replace('Current-root padding dispatcher', LABELS['candidate'])
    assert (TOOLS/'audit.py').read_text() == expected
    return compatible


def prepare():
    assert not BASE.exists(); compatible = previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source, name):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in UNCHANGED+['remote_prepare.py']: copy(TOOLS/name, 'tools/'+name)
    copy(TOOLS/'README.md', 'prospective-models.md')
    shutil.copyfile(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    (bundle/'evidence').mkdir(); save(bundle/'evidence/compatibility.json', compatible)
    old = read(CURRENT/'payload.json'); receipt = read(CURRENT/'collected/collection.json'); links = {}
    for name, wanted in old['files'].items():
        if name.startswith(('assets/', 'parakeet-reference/', 'runtimes/candidate/')):
            assert pin(CURRENT/'collected'/name) == wanted == receipt['files'][name], name
        if name.startswith(('assets/', 'parakeet-reference/')): links[name] = dict(source=REMOTE_CURRENT+'/'+name, identity=wanted)
    products = {role: compatible[role] for role in LABELS}
    for role in LABELS:
        for name, wanted in old['files'].items():
            if name.startswith('runtimes/candidate/'):
                suffix = name.removeprefix('runtimes/candidate/')
                links['runtimes/'+role+'/'+suffix] = dict(source=REMOTE_CURRENT+'/'+name, identity=wanted)
        for name, wanted in products[role].items():
            remote = REMOTE_QUALIFIED+'/runtime' if role == 'selected' else REMOTE_FIRST+'/runtimes/candidate'
            links['runtimes/'+role+'/'+name] = dict(source=remote+'/'+name, identity=wanted)
    terminals = []
    for folder, label, remote, suffix, remote_receipt in [
        (CURRENT, 'models', REMOTE_CURRENT, 'collected/collection.json', 'collection.json'),
        (QUALIFIED, 'root', REMOTE_QUALIFIED, 'collected/collection.json', 'collection.json'),
        (CONTRACTS, 'contracts', '/dev/shm/lokad-decrow3-20260927', 'collected/collection.json', 'collection.json'),
        (SCREEN, 'screen', '/dev/shm/lokad-decrow-screen2-20260927', 'capture-collected/capture-collection.json', 'capture-collection.json'),
        (DIAGNOSIS, 'diagnosis', '/dev/shm/lokad-unmapped-calls-20260927', 'capture-collected/capture-collection.json', 'capture-collection.json')]:
        copy(folder/'closed.json', 'evidence/'+label+'-closed.json')
        copy(folder/'analysis.json', 'evidence/'+label+'-analysis.json')
        receipt_path = folder/suffix
        assert pin(receipt_path) == read(folder/'closed.json')['files'][suffix]
        target = 'evidence/'+label+'-collection.json'; copy(receipt_path, target)
        terminals.append(dict(remote=remote+'/'+remote_receipt, local=target))
    copy(CURRENT/'collected/manifests/candidate-parakeet.json', 'evidence/original-manifest.json')
    stage = dict(passed=True, identities=products, consumers=compatible['consumers'], links=links, labels=LABELS,
        terminals=terminals, external=old['external'], interpreter=old['interpreter'],
        previous_owner=read(DIAGNOSIS/'capture-collected/capture-collection.json')['identities'][0],
        release_admitted=False, failed_release_controls=compatible['failed_component_controls'],
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    for folder in [TOOLS, PARENT, PAD]:
        for p in folder.iterdir():
            if p.is_file():
                if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
                originals[p.relative_to(ROOT).as_posix()] = pin(p)
    originals.update(compatible['inputs'])
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), links=len(links), identities=products)))


if __name__ == '__main__': prepare()
