"""Freeze original complete-model checks for one unchanged LSTM layout."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from compatibility import ROOT, CURRENT, QUALIFIED, CONTRACTS, CONTROL, SCREEN, CAPTURE, DIAGNOSIS, PROOFS, review

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'observed-dense-where-models-amd'
PAD = TOOLS.parent/'pad-current-models-amd'
PREVIOUS = CURRENT
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-models-amd-20260927'
REMOTES = {
    CURRENT: '/dev/shm/lokad-parakeet-decoder-packed-row-models-20260927',
    QUALIFIED: '/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927',
    CONTRACTS: '/dev/shm/lokad-lstmlayout3-20260927',
    CONTROL: '/dev/shm/lokad-lstmlayout-baseline-20260927',
    SCREEN: '/dev/shm/lokad-lstmlayout-timing-20260927',
    CAPTURE: '/dev/shm/lokad-lstm-runtime-20260927'}
LABELS = dict(selected='Qualified current root', candidate='Prepared LSTM grouped weights')
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
        if name.startswith(('assets/', 'parakeet-reference/')):
            links[name] = dict(source=REMOTES[CURRENT]+'/'+name, identity=wanted)
    products = {role: compatible[role] for role in LABELS}
    for role in LABELS:
        for name, wanted in old['files'].items():
            if name.startswith('runtimes/candidate/'):
                suffix = name.removeprefix('runtimes/candidate/')
                links['runtimes/'+role+'/'+suffix] = dict(source=REMOTES[CURRENT]+'/'+name, identity=wanted)
        for name, wanted in products[role].items():
            remote = REMOTES[QUALIFIED]+'/runtime' if role == 'selected' else REMOTES[CONTRACTS]+'/runtimes/candidate'
            links['runtimes/'+role+'/'+name] = dict(source=remote+'/'+name, identity=wanted)
    terminals = []
    for folder, label, filename, digest in PROOFS:
        proof = read(folder/filename)
        copy(folder/filename, 'evidence/'+label+'-proof.json')
        if (folder/'analysis.json').exists(): copy(folder/'analysis.json', 'evidence/'+label+'-analysis.json')
        if folder not in REMOTES: continue
        receipt_path = folder/'collected/collection.json'
        assert pin(receipt_path) == proof['files']['collected/collection.json']
        target = 'evidence/'+label+'-collection.json'; copy(receipt_path, target)
        terminals.append(dict(remote=REMOTES[folder]+'/collection.json', local=target, code=1 if folder == CONTRACTS else 0))
    copy(CURRENT/'collected/manifests/candidate-parakeet.json', 'evidence/original-manifest.json')
    stage = dict(passed=True, identities=products, consumers=compatible['consumers'], links=links, labels=LABELS,
        terminals=terminals, external=old['external'], interpreter=old['interpreter'],
        previous_owner=read(CAPTURE/'collected/collection.json')['identities'][0],
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
