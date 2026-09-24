"""Freeze complete Pyannote regression timing and long-meeting qualification."""
import ast,importlib.util,json,shutil,sys,tarfile
from pathlib import Path
from protocol import pin,read,save
from checks import prereqs
from consumer_scope import verify_scope
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924'
OLD=ROOT/'artifacts/parakeet-validated-composition-pyannote-app-amd-20260924';APP_PAYLOAD=OLD/'collected'
PARAKEET_APP=ROOT/'artifacts/parakeet-observed-dense-where-app-amd-20260924'
GRAPHS=ROOT/'artifacts/parakeet-observed-dense-where-graphs-amd-20260924'
PRIOR=dict(product=ROOT/'artifacts/parakeet-observed-dense-where-models-amd-20260924',
    models=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-amd-20260924',
    parakeet=ROOT/'artifacts/parakeet-observed-dense-where-models-amd-20260924',
    shared=ROOT/'artifacts/parakeet-observed-dense-where-shared-amd-20260924')
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('application_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)

def previous_closed():
    verify_scope()
    assert pin(OLD/'closed.json')['sha256']=='73e94016cf7c096c1bb9da9775459a5190e8ad60782c9faa1e7596f26cf4b3d1'
    assert read(PARAKEET_APP/'closed.json')['admitted']
    assert read(GRAPHS/'closed.json')['admitted'] and read(GRAPHS/'closed.json')['all_controls_passed']
    identities=read(PRIOR['models']/'analysis.json')['identities']
    graph=read(GRAPHS/'payload.json')['products'];app=read(PARAKEET_APP/'analysis.json')['identities']
    for role,label in [('current','selected'),('candidate','candidate')]:
        assert graph[role]['Lokad.Onnx.dll']==identities[label]['Lokad.Onnx.dll']
        assert app[role]==identities[label]
    assert read(PRIOR['product']/'closed.json')['analysis']==pin(PRIOR['product']/'analysis.json')
    assert read(PRIOR['parakeet']/'closed.json')['analysis']==pin(PRIOR['parakeet']/'analysis.json')
    for folder in [*PRIOR.values(),OLD,PARAKEET_APP,GRAPHS]:
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name

def prepare():
    previous_closed();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals=verify_scope();prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','meeting_protocol.py','meetings_audit.py','admission.py','semantics.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'graph_prerequisite.py',bundle/'tools/graph_prerequisite.py')
    for name in ['closed.json','analysis.json']:copy(GRAPHS/name,bundle/'evidence/graph-qualification'/name)
    for label,folder in [*PRIOR.items(),('baseline',OLD),('parakeet-app',PARAKEET_APP),('graphs',GRAPHS)]:
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
        if label in PRIOR:prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for family in ['pyannote','parakeet']:
        copy(APP_PAYLOAD/'manifests'/('candidate-'+family+'.json'),bundle/'evidence'/('original-'+family+'.json'))
    copy(APP_PAYLOAD/'meetings/manifest.json',bundle/'evidence/original-meetings.json')
    copy(APP_PAYLOAD/'meetings-run/output/result.json',bundle/'evidence/selected-meetings.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    stage=dict(passed=True,identities=read(PRIOR['models']/'analysis.json')['identities'],prerequisites=prerequisites,
        graph_qualification=dict(closed=pin(GRAPHS/'closed.json'),analysis=pin(GRAPHS/'analysis.json')),
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
