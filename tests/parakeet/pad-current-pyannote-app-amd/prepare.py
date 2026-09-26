"""Freeze the unchanged Pyannote application after fresh graph admission."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save
from checks import prereqs
from consumer_scope import verify_scope
from graph_prerequisite import validate as validate_graph

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-pad-current-pyannote-app-amd-20260926'
OLD=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924'
APP_PAYLOAD=OLD/'collected'
GRAPHS=ROOT/'artifacts/parakeet-pad-current-graphs-v2-amd-20260926'
PRIOR=dict(models=ROOT/'artifacts/parakeet-pad-current-pyannote-amd-20260926',
    parakeet=ROOT/'artifacts/parakeet-pad-current-models-amd-20260926',
    shared=ROOT/'artifacts/parakeet-pad-current-shared-amd-20260926',
    **{'parakeet-app':ROOT/'artifacts/parakeet-pad-current-app-amd-20260926'},baseline=OLD)
DIGESTS=dict(models='5e2b54bb685d9e954d3840524c486b55a22aadc5a574b6cc189bedf5bf487435',
    parakeet='194eb9a48b64df22d19b51adfc3ebe3accc56004544620123c128c2a76f066bf',
    shared='d1cb211096aa6a02ea70601b93cb615186c5d3b00472dced1c53283a1a8d3e3c',
    **{'parakeet-app':'2b40b54d8e94de7326ceec5228ad1529c7513964211e8b5dfbb2c921cd798151'},
    baseline='5dfd12f296a49b40e8731d77289fc94198df9ab2fc64a7e0252d7b3d8a2af4b0')
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
loader=importlib.util.spec_from_file_location('application_monitor',MONITOR)
monitor=importlib.util.module_from_spec(loader);loader.loader.exec_module(monitor)


def previous_closed():
    verify_scope()
    for label,folder in PRIOR.items():
        assert pin(folder/'closed.json')['sha256']==DIGESTS[label]
    for folder in [*PRIOR.values(),GRAPHS]:
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(PRIOR['parakeet-app']/'closed.json')['admitted']
    pair=read(PRIOR['models']/'analysis.json')['identities']
    products={role:{'Lokad.Onnx.dll':pair[label]['Lokad.Onnx.dll']} for role,label in [('current','selected'),('candidate','candidate')]}
    validate_graph(read(GRAPHS/'analysis.json'),read(GRAPHS/'closed.json'),products)
    from prerequisites import validate
    validate({label:read(folder/'analysis.json') for label,folder in PRIOR.items()},
             dict(identities=pair,consumers=read(OLD/'analysis.json')['consumers']))


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    originals=verify_scope();prerequisites={}
    def copy(source,target):
        target=bundle/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','prerequisites.py',
                 'graph_prerequisite.py','meeting_protocol.py','meetings_audit.py','admission.py','semantics.py']:
        copy(TOOLS/name,'tools/'+name)
    for name in ['closed.json','analysis.json','payload.json']:copy(GRAPHS/name,'evidence/graph-qualification/'+name)
    copy(GRAPHS/'collected/collection.json','evidence/graph-qualification/collection.json')
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,'evidence/'+label+'/'+name)
        copy(folder/'collected/collection.json','evidence/'+label+'/collection.json')
        prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for family in ['pyannote','parakeet']:
        copy(APP_PAYLOAD/'manifests'/('candidate-'+family+'.json'),'evidence/original-'+family+'.json')
    copy(APP_PAYLOAD/'meetings/manifest.json','evidence/original-meetings.json')
    copy(APP_PAYLOAD/'meetings-run/output/result.json','evidence/selected-meetings.json')
    copy(TOOLS/'README.md','prospective-plan.md')
    stage=dict(passed=True,identities=read(PRIOR['models']/'analysis.json')['identities'],prerequisites=prerequisites,
        graph_qualification=dict(closed=pin(GRAPHS/'closed.json'),analysis=pin(GRAPHS/'analysis.json')),
        consumers=read(OLD/'analysis.json')['consumers'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    prereqs(bundle,stage);save(bundle/'stage.json',stage)
    for path in [*TOOLS.iterdir(),MONITOR]:
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
