"""Freeze full-model qualification of the fixed rational sigmoid pair."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save
from compatibility import ROOT,CURRENT,QUALIFIED,BUILD,SCREEN,MEMORY,review

TOOLS=Path(__file__).resolve().parent
PARENT=TOOLS.parent/'observed-dense-where-models-amd'
PREVIOUS=CURRENT
BASE=ROOT/'artifacts/parakeet-rational-sigmoid-models-amd-20260927'
REMOTE_CURRENT='/dev/shm/lokad-parakeet-pad-current-models-20260926'
REMOTE_QUALIFIED='/dev/shm/lokad-parakeet-pad-current-root-20260926'
REMOTE_BUILD='/dev/shm/lokad-parakeet-rational-sigmoid-build-20260927'
LABELS=dict(selected='Qualified current root',candidate='ORT-derived rational sigmoid')
UNCHANGED=['protocol.py','remote.py','native_audit.py','public_audit.py']
PAD=TOOLS.parent/'pad-current-models-amd'


def previous_closed():
    compatible=review()
    for name in UNCHANGED:assert pin(TOOLS/name)==pin(PARENT/name),name
    expected=(PARENT/'audit.py').read_text().replace('M70 current release',LABELS['selected']).replace('M70 observed-mask composition',LABELS['candidate'])
    assert (TOOLS/'audit.py').read_text()==expected
    original=(PAD/'checks.py').read_text()
    start=original.index('def exact_native(');end=original.index('\n\ndef qualify(',start)
    replacement="def compare_native(base, isa):\n    from cross_numeric import compare_native as compare\n    return compare(base, isa)\n"
    expected=original[:start]+replacement+original[end:]
    expected=expected.replace('complete exact same-platform results','complete same-platform results with bounded float differences')
    expected=expected.replace("report['exact_selected_comparisons'] = exact_native(base, isa)","report['selected_comparisons'] = compare_native(base, isa)")
    assert (TOOLS/'checks.py').read_text()==expected
    for name,wanted in read(TOOLS/'dependencies.json').items():assert pin(ROOT/name)==wanted,name
    return compatible


def prepare():
    compatible=previous_closed();assert not BASE.exists()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,name):
        target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in UNCHANGED+['checks.py','cross_numeric.py','remote_prepare.py']:copy(TOOLS/name,'tools/'+name)
    copy(TOOLS/'README.md','prospective-models.md')
    shutil.copy2(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    (bundle/'evidence').mkdir(exist_ok=True);save(bundle/'evidence/compatibility.json',compatible)
    old=read(CURRENT/'payload.json');receipt=read(CURRENT/'collected/collection.json');links={}
    for name,wanted in old['files'].items():
        if name.startswith(('assets/','parakeet-reference/','runtimes/candidate/')):
            assert pin(CURRENT/'collected'/name)==wanted==receipt['files'][name],name
        if name.startswith(('assets/','parakeet-reference/')):
            links[name]=dict(source=REMOTE_CURRENT+'/'+name,identity=wanted)
    products={role:compatible[role] for role in LABELS}
    for role in LABELS:
        for name,wanted in old['files'].items():
            if name.startswith('runtimes/candidate/'):
                suffix=name.removeprefix('runtimes/candidate/')
                links['runtimes/'+role+'/'+suffix]=dict(source=REMOTE_CURRENT+'/'+name,identity=wanted)
        for name,wanted in products[role].items():
            remote=REMOTE_QUALIFIED if role=='selected' else REMOTE_BUILD
            links['runtimes/'+role+'/'+name]=dict(source=remote+'/runtime/'+name,identity=wanted)
    terminals=[]
    folders=[(CURRENT,'models',REMOTE_CURRENT,'collected/collection.json','collection.json'),
             (QUALIFIED,'root',REMOTE_QUALIFIED,'collected/collection.json','collection.json'),
             (BUILD,'build',REMOTE_BUILD,'capture-collected/capture-collection.json','capture-collection.json'),
             (SCREEN,'screen','/dev/shm/lokad-parakeet-rational-sigmoid-screen-20260927','capture-collected/capture-collection.json','capture-collection.json'),
             (MEMORY,'memory','/dev/shm/lokad-parakeet-rational-sigmoid-fallback-diagnostic-20260927','capture-collected/capture-collection.json','capture-collection.json')]
    for folder,label,remote,local_receipt,remote_receipt in folders:
        copy(folder/'closed.json','evidence/'+label+'-closed.json')
        if label!='memory':copy(folder/'analysis.json','evidence/'+label+'-analysis.json')
        receipt_path=folder/local_receipt
        assert pin(receipt_path)==read(folder/'closed.json')['files'][local_receipt]
        target='evidence/'+label+'-collection.json';copy(receipt_path,target)
        terminals.append(dict(remote=remote+'/'+remote_receipt,local=target))
    copy(CURRENT/'collected/manifests/candidate-parakeet.json','evidence/original-manifest.json')
    stage=dict(passed=True,identities=products,consumers=compatible['consumers'],links=links,labels=LABELS,
        terminals=terminals,external=old['external'],interpreter=old['interpreter'],
        previous_owner=read(MEMORY/'capture-collected/capture-collection.json')['identities'][0],
        release_admitted=False,failed_release_controls=compatible['failed_component_controls'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for folder in [TOOLS,PARENT,PAD]:
        for p in folder.iterdir():
            if p.is_file():
                if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),str(p))
                originals[p.relative_to(ROOT).as_posix()]=pin(p)
    for name,wanted in compatible['inputs'].items():originals[name]=wanted
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),links=len(links),identities=products)))


if __name__=='__main__':prepare()
