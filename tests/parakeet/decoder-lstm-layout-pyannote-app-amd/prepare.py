"""Reuse complete Pyannote application checks for the fixed LSTM layout pair."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import sys
PARENT=Path(__file__).resolve().parent.parent/'pad-current-pyannote-app-amd'
sys.path.insert(1,str(PARENT))
from protocol import pin,read,save
from checks import prereqs
from consumer_scope import verify_scope
from graph_prerequisite import validate as validate_graph

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-decoder-lstm-layout-pyannote-app-amd-20260927'
OLD=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924'
APP_PAYLOAD=OLD/'collected'
GRAPHS=ROOT/'artifacts/parakeet-decoder-lstm-layout-graphs-amd-20260927'
GRAPH_DIGEST=None  # Bind the actual new graph closure after its original audit passes.
PRIOR=dict(models=ROOT/'artifacts/parakeet-decoder-lstm-layout-pyannote-amd-20260927',
    parakeet=ROOT/'artifacts/parakeet-decoder-lstm-layout-models-amd-20260927',
    shared=ROOT/'artifacts/parakeet-decoder-lstm-layout-shared-amd-20260927',
    **{'parakeet-app':ROOT/'artifacts/parakeet-decoder-lstm-layout-app-amd-20260927'},baseline=OLD)
DIGESTS=dict(models='8402f31da2de3034f8f64a6ed30644b4fb0e527c72df77b2c2c04113c747e808',
    parakeet='1acb1d884eb5293e0487e1b34d8c842b7240823040e63d6dc2a043ec89f7a111',
    shared='dd317fe93e509cc9575237b73e5f9c44cb6b44e51412bc9431dedd9e7fd0b25d',
    **{'parakeet-app':'e3ba182a323209926e26c885d190875b04087d430f5e97bed41f460031b4883b'},
    baseline='5dfd12f296a49b40e8731d77289fc94198df9ab2fc64a7e0252d7b3d8a2af4b0')
QUALIFIED=ROOT/'artifacts/parakeet-decoder-packed-row-pyannote-app-amd-20260927'
COMPATIBLE=PRIOR['parakeet']/'collected/evidence/compatibility.json'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
loader=importlib.util.spec_from_file_location('application_monitor',MONITOR)
monitor=importlib.util.module_from_spec(loader);loader.loader.exec_module(monitor)


def previous_closed():
    verify_scope()
    assert GRAPH_DIGEST is not None, 'Require the actual admitted graph closure before preparation'
    assert pin(GRAPHS/'closed.json')['sha256']==GRAPH_DIGEST
    for label,folder in PRIOR.items():
        assert pin(folder/'closed.json')['sha256']==DIGESTS[label]
    assert pin(QUALIFIED/'closed.json')['sha256']=='17861b3815ecb72d636fc86562e2c8605d01af9cc561841f069d5d1c23e14188'
    for folder in [*PRIOR.values(),GRAPHS,QUALIFIED]:
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(PRIOR['parakeet-app']/'closed.json')['admitted']
    pair=read(PRIOR['models']/'analysis.json')['identities']
    products={role:{'Lokad.Onnx.dll':pair[label]['Lokad.Onnx.dll']} for role,label in [('current','selected'),('candidate','candidate')]}
    validate_graph(read(GRAPHS/'analysis.json'),read(GRAPHS/'closed.json'),products)
    from prerequisites import validate
    validate({label:read(folder/'analysis.json') for label,folder in PRIOR.items()},
             dict(identities=pair,consumers=read(OLD/'analysis.json')['consumers']),
             read(COMPATIBLE),read(QUALIFIED/'analysis.json'))


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    originals=verify_scope();prerequisites={}
    def copy(source,target):
        target=bundle/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','prerequisites.py',
                 'graph_prerequisite.py','meeting_protocol.py','meetings_audit.py','admission.py','semantics.py']:
        copy((TOOLS if name in ['remote_prepare.py','prerequisites.py'] else PARENT)/name,'tools/'+name)
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
    copy(COMPATIBLE,'evidence/product-compatibility.json')
    for name in ['closed.json','analysis.json']:
        copy(QUALIFIED/name,'evidence/consumer-qualification/'+name)
    copy(TOOLS/'README.md','prospective-plan.md')
    stage=dict(passed=True,identities=read(PRIOR['models']/'analysis.json')['identities'],prerequisites=prerequisites,
        graph_qualification=dict(closed=pin(GRAPHS/'closed.json'),analysis=pin(GRAPHS/'analysis.json')),
        consumers=read(OLD/'analysis.json')['consumers'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    prereqs(bundle,stage);save(bundle/'stage.json',stage)
    for path in [*TOOLS.iterdir(),*PARENT.iterdir(),MONITOR]:
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
