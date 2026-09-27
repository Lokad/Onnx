"""Freeze the actual integrated root only after every fixed pointwise remainder release prerequisite."""
import ast
import json
import shutil
import tarfile
from source_scope import ROOT,TOOLS,PARENT,APP,SOURCE,BUILD,FIXTURE,QUALIFIED,APPLIED,verify_source,root_files,verify_root,load
from protocol import pin,read,save
from consumer_scope import verify_scope
from prerequisites import validate,verify
from graph_prerequisite import validate as validate_graph

BASE=ROOT/'artifacts/parakeet-pointwise-tail-root-amd-20260927'
CONTRACTS=ROOT/'artifacts/parakeet-pointwise-tail-arithmetic-contracts-amd-20260927'
CONSUMER_QUALIFIED=ROOT/'artifacts/parakeet-decoder-lstm-layout-pyannote-app-amd-20260927'
GRAPH=ROOT/'artifacts/parakeet-pointwise-tail-graphs-amd-20260927'
PRIOR=dict(models=ROOT/'artifacts/parakeet-pointwise-tail-pyannote-amd-20260927',
    parakeet=ROOT/'artifacts/parakeet-pointwise-tail-models-amd-20260927',
    shared=ROOT/'artifacts/parakeet-pointwise-tail-shared-amd-20260927',
    baseline=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924',
    **{'qualified-root':QUALIFIED,'consumer-qualified':CONSUMER_QUALIFIED,
    'parakeet-app':ROOT/'artifacts/parakeet-pointwise-tail-app-amd-20260927',
    'pyannote-app':ROOT/'artifacts/parakeet-pointwise-tail-pyannote-app-amd-20260927'})
DIGESTS=dict(models='f284cc72de08b0113daa8a5482e2a2dbea2bd6a89ea00e128a569036394eebb8',
    parakeet='f773277daa848121c199c58e67bce3777677c2ee2654fbe26250d02719155960',
    shared='fe35674de649414c520cc73812d485c93cde6aadbc05db0c17ab94ca883e8c1b',
    baseline='5dfd12f296a49b40e8731d77289fc94198df9ab2fc64a7e0252d7b3d8a2af4b0',
    **{'consumer-qualified':'5942b121dadb101c52e4221b7e1143d0c7ac37719d0de8db1e64f54f990e1404',
       'qualified-root':'efb99eea455647c64bbe0811c1b7d46387add59049ac81e48a926b250e7c42da',
       'parakeet-app':'505da6ab8c9ce1d61b6f2b241f2bbf45d156bd8c621e40bb2caed4d6967281a3'})
GRAPH_DIGEST=None
PYANNOTE_APP_DIGEST=None
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
    assert GRAPH_DIGEST is not None and PYANNOTE_APP_DIGEST is not None, 'Require actual admitted graph and complete Pyannote application closures'
    assert pin(GRAPH/'closed.json')['sha256']==GRAPH_DIGEST
    assert pin(PRIOR['pyannote-app']/'closed.json')['sha256']==PYANNOTE_APP_DIGEST
    for label,folder in PRIOR.items():
        if label in DIGESTS:assert pin(folder/'closed.json')['sha256']==DIGESTS[label],label
    for folder in [*PRIOR.values(),GRAPH]:
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert pin(CONTRACTS/'closed.json')['sha256']=='558f2a523febd0d794bc7da6cdae3381513b7ff3d75dfd4ca4f133f618821cc8'
    proof=read(CONTRACTS/'closed.json');assert proof['completed'] and proof['arithmetic_contract_passed']
    for name,wanted in proof['files'].items():assert pin(CONTRACTS/name)==wanted,name
    assert pin(CONTRACTS/'codegen-review.json')['sha256']=='3715408c95a3164b2f06e36db4aeb2bea95f08ee8f7afb03cbb38e9747d9d692'
    for label in ['parakeet-app','pyannote-app']:assert read(PRIOR[label]/'closed.json')['admitted']
    spec=evidence_spec()
    validate({k:read(p/'analysis.json') for k,p in PRIOR.items()},spec,read(COMPATIBLE),
             read(CONTRACTS/'analysis.json'),read(BUILD/'build-review.json'),read(CONTRACTS/'codegen-review.json'))
    products={role:{'Lokad.Onnx.dll':spec['identities'][label]['Lokad.Onnx.dll']}
              for role,label in [('current','selected'),('candidate','candidate')]}
    validate_graph(read(GRAPH/'analysis.json'),read(GRAPH/'closed.json'),products)
    return source


def previous_closed():
    source=gates();value=read(APPLIED/'applied.json');spec=evidence_spec()
    assert value['passed'] and value['source_files']==root_files(source)
    assert value['prepared']==spec['source_prepared'] and value['fixture']==pin(FIXTURE)
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
    copy(BUILD/'closed.json','evidence/contracts-failure/closed.json')
    copy(BUILD/'build-review.json','evidence/contracts-build.json')
    copy(CONTRACTS/'codegen-review.json','evidence/contracts-codegen.json')
    copy(COMPATIBLE,'evidence/product-compatibility.json')
    copy(APPLIED/'applied.json','evidence/root-applied.json')
    copy(SOURCE/'prepared.json','evidence/source-prepared.json')
    copy(FIXTURE,'evidence/PackedColumnRemainderTests.cs')
    copy(TOOLS/'README.md','prospective-plan.md')
    stage=dict(passed=True,**evidence_spec(),root_integration=pin(APPLIED/'applied.json'),
        source_scope='445 root inputs; all3288 Core/697 Data methods and metadata equal the measured pointwise candidate; two independent portable remainder facts; source policy unchanged.',
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
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=445)))


if __name__=='__main__':prepare()
