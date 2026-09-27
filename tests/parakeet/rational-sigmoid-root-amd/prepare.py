"""Freeze the actual integrated root only after every fixed rational sigmoid release prerequisite."""
import ast
import json
import shutil
import tarfile
from source_scope import ROOT,TOOLS,PARENT,APP,SOURCE,FIXTURE,QUALIFIED,APPLIED,verify_source,root_files,verify_root,load
from protocol import pin,read,save
from consumer_scope import verify_scope
from prerequisites import validate,verify
from graph_prerequisite import validate as validate_graph

BASE=ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
CONTRACTS=ROOT/'artifacts/parakeet-rational-sigmoid-build-amd-20260927'
CONSUMER_QUALIFIED=ROOT/'artifacts/parakeet-pad-current-pyannote-app-amd-20260926'
GRAPH=ROOT/'artifacts/parakeet-rational-sigmoid-graphs-amd-20260927'
PRIOR=dict(models=ROOT/'artifacts/parakeet-rational-sigmoid-pyannote-amd-20260927',
    parakeet=ROOT/'artifacts/parakeet-rational-sigmoid-models-amd-20260927',
    shared=ROOT/'artifacts/parakeet-rational-sigmoid-shared-amd-20260927',
    baseline=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924',
    **{'qualified-root':QUALIFIED,'consumer-qualified':CONSUMER_QUALIFIED,
    'parakeet-app':ROOT/'artifacts/parakeet-rational-sigmoid-app-amd-20260927',
    'pyannote-app':ROOT/'artifacts/parakeet-rational-sigmoid-pyannote-app-amd-20260927'})
DIGESTS=dict(models='41333fc2a2c60c45596ebb87d65c4ce5fdfa23108ed2064a2db5e308dc9e5138',
    parakeet='3d2e01b5aca3f66c434d5d1e8124c2b0d08327bc14dd37409c0e1884619769d8',
    shared='86ad6b3f5d11549dc968270e6ef5936787a266a84969de1932e750450ee6da65',
    baseline='5dfd12f296a49b40e8731d77289fc94198df9ab2fc64a7e0252d7b3d8a2af4b0',
    **{'consumer-qualified':'60ea42f647c8c55b5ea97c289a69ccefd04e8b6cdb645c12daf996611e3fb7e8',
       'qualified-root':'71c80efd687562cba2e2b5d03e9e036b93de09d0a30f74b976b86355ed12fdf0',
       'parakeet-app':'629183e44963638c0285ea4daf70f8f9aab3b94d5e4c6164225d205aaaa443b4'})
COMPATIBLE=PRIOR['parakeet']/'collected/evidence/compatibility.json'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
monitor=load('padding_root_monitor',MONITOR)


def evidence_spec():
    pair=read(PRIOR['models']/'analysis.json')['identities']
    return dict(identities=pair,measured=pair['candidate'],source_prepared=pin(SOURCE/'prepared.json'),
        consumers=read(PRIOR['baseline']/'analysis.json')['consumers'],
        prerequisites={k:dict(closed=pin(p/'closed.json'),analysis=pin(p/'analysis.json')) for k,p in PRIOR.items()},
        graph_qualification=dict(closed=pin(GRAPH/'closed.json'),analysis=pin(GRAPH/'analysis.json')))


def gates():
    verify_scope();source=verify_source()
    for label,folder in PRIOR.items():
        if label in DIGESTS:assert pin(folder/'closed.json')['sha256']==DIGESTS[label],label
    assert pin(CONTRACTS/'closed.json')['sha256']=='3301ae58b42e1fd51435f54191cf4bba42c6cbf8e2ad6bf4279af67ae5d3d89a'
    for folder in [*PRIOR.values(),GRAPH,CONTRACTS]:
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    for label in ['parakeet-app','pyannote-app']:assert read(PRIOR[label]/'closed.json')['admitted']
    spec=evidence_spec()
    validate({k:read(p/'analysis.json') for k,p in PRIOR.items()},spec,read(COMPATIBLE),
             read(CONTRACTS/'analysis.json'),read(CONTRACTS/'build-review.json'))
    products={role:{'Lokad.Onnx.dll':spec['identities'][label]['Lokad.Onnx.dll']}
              for role,label in [('current','selected'),('candidate','candidate')]}
    validate_graph(read(GRAPH/'analysis.json'),read(GRAPH/'closed.json'),products)
    return source


def previous_closed():
    source=gates();value=read(APPLIED/'applied.json');spec=evidence_spec()
    assert value['passed'] and value['source_files']==root_files(source)
    assert value['prepared']==spec['source_prepared'] and value['fixture']==pin(FIXTURE/'prepared.json')
    assert value['prerequisites']=={k:v['closed'] for k,v in spec['prerequisites'].items()}
    assert value['graph_qualification']==spec['graph_qualification']['closed']
    verify_root(value['source_files'])


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    originals=verify_scope()
    def copy(source,target):
        target=bundle/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in read(APPLIED/'applied.json')['source_files']:copy(ROOT/name,'source/'+name)
    for p in (QUALIFIED/'bundle/consumer').iterdir():
        if p.is_file():copy(p,'consumer/'+p.name)
    copy(TOOLS.parent/'owned-batch-isolation-build/Bridge.cs.txt','bridge-source/Program.cs')
    copy(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj','bridge-source/Bridge.csproj')
    copy(ROOT/'global.json','bridge-source/global.json')
    copy(QUALIFIED/'bundle/evidence/tensor-source.tar','evidence/tensor-source.tar')
    for mode in ['', '-256']:
        for name in ['backend','tensors']:
            copy(QUALIFIED/'collected'/(name+'-tests'+mode)/(name+'.trx'),'evidence/selected-'+name+mode+'.trx')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','new_cases.py',
                 'graph_prerequisite.py','model_prerequisites.py','prerequisites.py','application_admission.py']:
        origin=APP/'prerequisites.py' if name=='model_prerequisites.py' else (TOOLS if name in ['remote_prepare.py','checks.py','new_cases.py','prerequisites.py'] else PARENT)/name
        copy(origin,'tools/'+name)
    for label,folder in [*PRIOR.items(),('graph-qualification',GRAPH)]:
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,'evidence/'+label+'/'+name)
        copy(folder/'collected/collection.json','evidence/'+label+'/collection.json')
    for name in ['closed.json','analysis.json','build-review.json']:copy(CONTRACTS/name,'evidence/contracts/'+name)
    copy(COMPATIBLE,'evidence/product-compatibility.json')
    copy(APPLIED/'applied.json','evidence/root-applied.json')
    copy(SOURCE/'prepared.json','evidence/source-prepared.json')
    copy(FIXTURE/'prepared.json','evidence/fixture-prepared.json')
    copy(TOOLS/'README.md','prospective-plan.md')
    stage=dict(passed=True,**evidence_spec(),root_integration=pin(APPLIED/'applied.json'),
        source_scope='439 root inputs; all3283 Core/697 Data bodies, flags and public/assembly metadata equal the measured rational sigmoid candidate; eight unchanged arithmetic facts, no source-policy exception.',
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    verify(bundle,stage);save(bundle/'stage.json',stage)
    for path in [*TOOLS.iterdir(),*PARENT.iterdir(),*APP.iterdir(),MONITOR]:
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=439)))


if __name__=='__main__':prepare()
