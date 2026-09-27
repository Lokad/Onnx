"""Freeze the actual integrated root only after every fixed prepared-row release prerequisite."""
import ast
import json
import shutil
import tarfile
from source_scope import ROOT,TOOLS,PARENT,APP,SOURCE,FIXTURE,QUALIFIED,APPLIED,verify_source,root_files,verify_root,load
from protocol import pin,read,save
from consumer_scope import verify_scope
from prerequisites import validate,verify
from graph_prerequisite import validate as validate_graph

BASE=ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
CONTRACTS=ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v3-amd-20260927'
CONSUMER_QUALIFIED=ROOT/'artifacts/parakeet-rational-sigmoid-pyannote-app-amd-20260927'
GRAPH=ROOT/'artifacts/parakeet-decoder-packed-row-graphs-amd-20260927'
PRIOR=dict(models=ROOT/'artifacts/parakeet-decoder-packed-row-pyannote-amd-20260927',
    parakeet=ROOT/'artifacts/parakeet-decoder-packed-row-models-amd-20260927',
    shared=ROOT/'artifacts/parakeet-decoder-packed-row-shared-amd-20260927',
    baseline=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924',
    **{'qualified-root':QUALIFIED,'consumer-qualified':CONSUMER_QUALIFIED,
    'parakeet-app':ROOT/'artifacts/parakeet-decoder-packed-row-app-amd-20260927',
    'pyannote-app':ROOT/'artifacts/parakeet-decoder-packed-row-pyannote-app-amd-20260927'})
DIGESTS=dict(models='74bd1f90716a11446846d12cbba344b501d2736777ac1d0b3eedd6eb10ca7a33',
    parakeet='5d6832083d103bef9db7bb733b1e96decbae22625f3138680ab6f9a426c78e3b',
    shared='bdd1c7664d902af7d135318f12856419a61d222e69bb073ae24e10b8aca8aca0',
    baseline='5dfd12f296a49b40e8731d77289fc94198df9ab2fc64a7e0252d7b3d8a2af4b0',
    **{'consumer-qualified':'f1c4e873f19f5562b2da8a31a59ec5b54f47959b8f4c1c996ec6c996f5e2218d',
       'qualified-root':'f6df53ce2ab773898bc84f144abeda97908d383fef9a4df4b7299a22a4d3594d',
       'parakeet-app':'26ac173f867a964b73761d3c716aa5db0c384909bb1849e2cc810ed90e0a1781'})
COMPATIBLE=PRIOR['parakeet']/'collected/evidence/compatibility.json'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
monitor=load('padding_root_monitor',MONITOR)


def evidence_spec():
    pair=read(PRIOR['models']/'analysis.json')['identities']
    return dict(identities=pair,measured=pair['candidate'],source_prepared=pin(SOURCE/'bundle/stage.json'),
        consumers=read(PRIOR['baseline']/'analysis.json')['consumers'],
        prerequisites={k:dict(closed=pin(p/'closed.json'),analysis=pin(p/'analysis.json')) for k,p in PRIOR.items()},
        graph_qualification=dict(closed=pin(GRAPH/'closed.json'),analysis=pin(GRAPH/'analysis.json')))


def gates():
    verify_scope();source=verify_source()
    for label,folder in PRIOR.items():
        if label in DIGESTS:assert pin(folder/'closed.json')['sha256']==DIGESTS[label],label
    assert pin(CONTRACTS/'closed.json')['sha256']=='fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d'
    for folder in [*PRIOR.values(),GRAPH,CONTRACTS]:
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    for label in ['parakeet-app','pyannote-app']:assert read(PRIOR[label]/'closed.json')['admitted']
    spec=evidence_spec()
    validate({k:read(p/'analysis.json') for k,p in PRIOR.items()},spec,read(COMPATIBLE),
             read(CONTRACTS/'analysis.json'),read(CONTRACTS/'analysis.json')['compiled'])
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
    for name in ['closed.json','analysis.json']:copy(CONTRACTS/name,'evidence/contracts/'+name)
    copy(COMPATIBLE,'evidence/product-compatibility.json')
    copy(APPLIED/'applied.json','evidence/root-applied.json')
    copy(SOURCE/'bundle/stage.json','evidence/source-prepared.json')
    copy(FIXTURE/'prepared.json','evidence/fixture-prepared.json')
    copy(TOOLS/'README.md','prospective-plan.md')
    stage=dict(passed=True,**evidence_spec(),root_integration=pin(APPLIED/'applied.json'),
        source_scope='441 root inputs; all3284 Core/697 Data methods and metadata equal the measured prepared-row candidate; four portable boundary/ownership facts; source policy unchanged.',
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
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=441)))


if __name__=='__main__':prepare()
