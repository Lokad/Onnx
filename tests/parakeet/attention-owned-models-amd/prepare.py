"""Freeze original full-model checks for the qualified attention preparation candidate."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile

TOOLS = Path(__file__).resolve().parent
LIB = TOOLS.parent/'decoder-lstm-layout-models-amd'
sys.path.insert(1,str(LIB))
from protocol import pin, read, save
from compatibility import ROOT, CURRENT, QUALIFIED, CONTRACTS, CENSUS, PROOFS, review

PARENT = TOOLS.parent/'observed-dense-where-models-amd'
PREVIOUS = CURRENT
BASE = ROOT/'artifacts/parakeet-attention-owned-models-amd-20260928'
REMOTES = {CURRENT:'/dev/shm/lokad-pwt-models-20260927',
    QUALIFIED:'/dev/shm/lokad-pwt-root-20260927',
    CONTRACTS:'/dev/shm/lokad-attention-owned-build-20260928',
    CENSUS:'/dev/shm/lokad-attention-owned-census-20260928'}
LABELS = dict(selected='Qualified current root', candidate='Owned square attention weights')
UNCHANGED = ['protocol.py','remote.py','checks.py','native_audit.py','public_audit.py']


def previous_closed():
    result = review()
    for name in UNCHANGED:
        assert pin(LIB/name) == pin(TOOLS.parent/'pad-current-models-amd'/name), name
    expected = (TOOLS.parent/'pad-current-models-amd/audit.py').read_text().replace('Current-root padding dispatcher','Prepared LSTM grouped weights')
    assert (LIB/'audit.py').read_text() == expected
    return result


def prepare():
    assert not BASE.exists(); compatible = previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source,name):
        target = bundle/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target); originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in UNCHANGED: copy(LIB/name,'tools/'+name)
    copy(TOOLS/'remote_prepare.py','tools/remote_prepare.py')
    copy(TOOLS/'README.md','prospective-models.md')
    shutil.copyfile(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    (bundle/'evidence').mkdir(); save(bundle/'evidence/compatibility.json',compatible)
    old = read(CURRENT/'payload.json'); receipt = read(CURRENT/'collected/collection.json'); links = {}
    for name,wanted in old['files'].items():
        if name.startswith(('assets/','parakeet-reference/','runtimes/candidate/')):
            assert pin(CURRENT/'collected'/name) == wanted == receipt['files'][name], name
        if name.startswith(('assets/','parakeet-reference/')):
            links[name] = dict(source=REMOTES[CURRENT]+'/'+name,identity=wanted)
    products = {role:compatible[role] for role in LABELS}
    for role in LABELS:
        for name,wanted in old['files'].items():
            if name.startswith('runtimes/candidate/'):
                suffix = name.removeprefix('runtimes/candidate/')
                links['runtimes/'+role+'/'+suffix] = dict(source=REMOTES[CURRENT]+'/'+name,identity=wanted)
        for name,wanted in products[role].items():
            runtime = compatible['selected_runtime'] if role == 'selected' else REMOTES[CONTRACTS]+'/runtime'
            links['runtimes/'+role+'/'+name] = dict(source=runtime+'/'+name,identity=wanted)
    terminals = []
    for folder,label,filename,digest in PROOFS:
        copy(folder/filename,'evidence/'+label+'-proof.json')
        copy(folder/'analysis.json','evidence/'+label+'-analysis.json')
        if folder not in REMOTES: continue
        path = 'capture-collected/capture-collection.json' if folder in [CONTRACTS,CENSUS] else 'collected/collection.json'
        receipt_path = folder/path
        assert pin(receipt_path) == read(folder/filename)['files'][path]
        target = 'evidence/'+label+'-collection.json'; copy(receipt_path,target)
        terminals.append(dict(remote=REMOTES[folder]+'/'+Path(path).name,local=target,code=0))
    copy(CONTRACTS/'build-review.json','evidence/build-review.json')
    copy(CURRENT/'collected/manifests/candidate-parakeet.json','evidence/original-manifest.json')
    stage = dict(passed=True,identities=products,consumers=compatible['consumers'],links=links,labels=LABELS,
        terminals=terminals,external=old['external'],interpreter=old['interpreter'],
        previous_owner=read(CENSUS/'capture-collected/capture-collection.json')['identities'][0],
        release_admitted=False,failed_release_controls=[],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for folder in [TOOLS,PARENT,LIB]:
        for p in folder.iterdir():
            if p.is_file():
                if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'),str(p))
                originals[p.relative_to(ROOT).as_posix()] = pin(p)
    originals.update(compatible['inputs'])
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),links=len(links),identities=products)))


if __name__ == '__main__': prepare()
