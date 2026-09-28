"""Freeze the actual integrated root only after every single transpose dispatch release prerequisite."""
import ast
import json
import shutil
import tarfile
from source_scope import ROOT,TOOLS,PARENT,APP,SOURCE,BUILD,FIXTURE,QUALIFIED,APPLIED,verify_source,root_files,verify_root,load
from protocol import pin,read,save
from consumer_scope import verify_scope
from prerequisites import validate,verify
from graph_prerequisite import validate as validate_graph

BASE=ROOT/'artifacts/parakeet-transpose-axis-root-amd-20260928'
CONTRACTS=ROOT/'artifacts/parakeet-transpose-axis-build-recovery-amd-20260928'
CONSUMER_QUALIFIED=ROOT/'artifacts/parakeet-attention-owned-pyannote-app-amd-20260928'
GRAPH=ROOT/'artifacts/parakeet-transpose-axis-graphs-amd-20260928'
PRIOR=dict(models=ROOT/'artifacts/parakeet-transpose-axis-pyannote-amd-20260928',
    parakeet=ROOT/'artifacts/parakeet-transpose-axis-models-amd-20260928',
    shared=ROOT/'artifacts/parakeet-transpose-axis-shared-amd-20260928',
    baseline=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924',
    **{'qualified-root':QUALIFIED,'consumer-qualified':CONSUMER_QUALIFIED,
    'parakeet-app':ROOT/'artifacts/parakeet-transpose-axis-app-amd-20260928',
    'pyannote-app':ROOT/'artifacts/parakeet-transpose-axis-pyannote-app-amd-20260928'})
DIGESTS=dict(models='3188186fc659d1846fccb3b6c251f7a91f24ecda641dff32ce052b98f1c972ca',
    parakeet='8568a2299a820273b7357eaaba9c151bc83121bfc781759afbe9142b8d7b5984',
    shared='fa74521cbe415ead594903a2b76649bcd4ec2edc20d075f616ae67948e920f1b',
    baseline='5dfd12f296a49b40e8731d77289fc94198df9ab2fc64a7e0252d7b3d8a2af4b0',
    **{'consumer-qualified':'d74cdd3d0aa6e15e9d24dba323eec0f30b38fd5ea7e1b011d2ec7cbad94922bf',
       'qualified-root':'ee4a38ff671cc3fd8cd608c0dc3008f5c1b99f61ba6c139f3deeaf0e9039305e',
       'parakeet-app':'2e4741a67772d71e4442630d7f9cf751dfba2a52e12631fffed3b2db62fefd85'})
GRAPH_DIGEST='147df8e4221d21fb80d2c9f69444b076c247a77e795b18e0d2973fab0bf941f5'
PYANNOTE_APP_DIGEST='a25671c2480b7b9db3ec03b9e24a97253cc75d9464715515237c26b785c36899'
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
    for folder,digest in [(CONTRACTS,'c0c063bb3f73eed2d8d7372d17dd1da3ce4f6a962e190ef101b504fb245e4ecd')]:
        proof=read(folder/'closed.json');assert proof['passed'] and pin(folder/'closed.json')['sha256']==digest
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    for label in ['parakeet-app','pyannote-app']:assert read(PRIOR[label]/'closed.json')['admitted']
    spec=evidence_spec()
    validate({k:read(p/'analysis.json') for k,p in PRIOR.items()},spec,read(COMPATIBLE),
             read(CONTRACTS/'analysis.json'),read(BUILD/'build-review.json'))
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
    copy(BUILD/'build-review.json','evidence/contracts-build.json')
    copy(COMPATIBLE,'evidence/product-compatibility.json')
    copy(APPLIED/'applied.json','evidence/root-applied.json')
    copy(SOURCE/'prepared.json','evidence/source-prepared.json')
    copy(FIXTURE,'evidence/TransposeAxisMovementTests.cs')
    copy(TOOLS/'README.md','prospective-plan.md')
    stage=dict(passed=True,**evidence_spec(),root_integration=pin(APPLIED/'applied.json'),
        source_scope='447 root inputs; all3288 Core/697 Data methods and metadata equal the measured transpose candidate; six portable layout/bit/ownership facts pass in both full modes; original source policy unchanged.',
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
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=447)))


if __name__=='__main__':prepare()
