"""Freeze complete Pyannote regression timing and long-meeting qualification."""
import ast,importlib.util,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
from checks import prereqs
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-first-use-kernels-pyannote-app-amd-20260923'
OLD=ROOT/'artifacts/pyannote-winograd-product-app-amd-20260923';APP_PAYLOAD=OLD/'collected'
PARAKEET_APP=ROOT/'artifacts/parakeet-first-use-kernels-app-amd-20260923'
PRIOR=dict(product=ROOT/'artifacts/parakeet-first-use-kernels-build-amd-20260923',
    models=ROOT/'artifacts/parakeet-first-use-kernels-pyannote-amd-20260923',
    parakeet=ROOT/'artifacts/parakeet-first-use-kernels-models-amd-v2-20260923',
    shared=ROOT/'artifacts/parakeet-first-use-kernels-shared-amd-20260923')
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('application_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)

def previous_closed():
    assert pin(OLD/'closed.json')['sha256']=='dc2c7b9f5086ab9b4ee615b9dad7643eaf4c1ed65c1cc71b3e76ee794237d88e'
    assert pin(PARAKEET_APP/'closed.json')['sha256']=='adb3a270880bb4f30ac87fec92c2f6e18495d56c8ed8cffcfb58a950f7a25128'
    assert read(PARAKEET_APP/'closed.json')['admitted']
    for folder in [*PRIOR.values(),OLD,PARAKEET_APP]:
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name

def prepare():
    previous_closed();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={};prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','meeting_protocol.py','meetings_audit.py','admission.py','semantics.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for label,folder in [*PRIOR.items(),('baseline',OLD),('parakeet-app',PARAKEET_APP)]:
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
        if label in PRIOR:prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for family in ['pyannote','parakeet']:
        copy(APP_PAYLOAD/'manifests'/('candidate-'+family+'.json'),bundle/'evidence'/('original-'+family+'.json'))
    copy(APP_PAYLOAD/'meetings/manifest.json',bundle/'evidence/original-meetings.json')
    copy(APP_PAYLOAD/'meetings-run/output/result.json',bundle/'evidence/selected-meetings.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    stage=dict(passed=True,identities=read(PRIOR['models']/'analysis.json')['identities'],prerequisites=prerequisites,
        consumers=read(OLD/'analysis.json')['consumers'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    prereqs(bundle,stage);save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))

if __name__=='__main__':prepare()
