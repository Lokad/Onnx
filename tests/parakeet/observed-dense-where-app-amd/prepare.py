"""Bind the unchanged complete application comparison to one diagnosed change."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-observed-dense-where-app-amd-20260924'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS=ROOT/'artifacts/parakeet-observed-dense-where-models-amd-20260924'
BUILD=ROOT/'artifacts/parakeet-observed-dense-where-inventory-amd-20260924'
NUMERICS=ROOT/'artifacts/parakeet-observed-dense-where-numerics-amd-20260924'
PROFILE=ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924'
FIRST_PROFILE=ROOT/'artifacts/parakeet-observed-dense-where-profile-amd-20260924'
QUALIFIED=ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924'
RELEASE_APP=ROOT/'artifacts/parakeet-validated-composition-app-amd-20260924'
SOURCE=ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'
PRIOR=dict(baseline=CURRENT,models=MODELS,build=BUILD,numerics=NUMERICS,profile=PROFILE,root=QUALIFIED,release_app=RELEASE_APP)
DIGESTS=dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='ff9a7a2b5f1156bd365d0f51564f1160e72a82f632ce6e544bb7f86a5663e337',
    build='7d5296b266feeacbf9c52bf57a689de2cdf0a839045130f15912912f5da1b7bd',
    numerics='0dfb5dcb4d5c5793c7c175837bde6f9469e791f62c02cf1b4df679ba4d81f0db',
    profile='8a6f509a210641650a9c057f472417dde7b4acc320f86eeb3d34ecbced4cc2e6',
    root='c7a1d2e11566e6eeeb965de6c9cedbf47df479fd51f194c797af412446281609',
    release_app='f04c09fbc6c0455c4420d6680d60bb4f8fc5cac9dda2f2507768ba94b86335e4')


def previous_closed():
    from consumer_scope import verify_scope
    assert verify_scope()
    for name,folder in PRIOR.items():
        proof=read(folder/'closed.json');assert proof['passed']
        if name in DIGESTS:assert pin(folder/'closed.json')['sha256']==DIGESTS[name]
        assert proof['analysis']==pin(folder/'analysis.json')
        root=ROOT if proof.get('paths_relative_to_repository') else folder
        for path,wanted in proof.get('files',{}).items():assert pin(root/path)==wanted,path
    proof=read(PROFILE/'closed.json')
    assert proof['audit_correction']==pin(PROFILE/'audit-correction.json')
    correction=read(PROFILE/'audit-correction.json');assert correction['passed']
    assert correction['corrected_auditor']==pin(TOOLS.parent/'observed-dense-where-results/audit_profile.py')
    assert proof['collection']==pin(PROFILE/'capture-collected/capture-collection.json')
    assert proof['transfer']==pin(PROFILE/'capture-transfer.json')
    for name,wanted in read(PROFILE/'capture-collected/capture-collection.json')['files'].items():
        assert pin(PROFILE/'capture-collected'/name)==wanted,name
    assert read(PROFILE/'analysis.json')['masking']['gain']>=.80
    first=proof['initial'];assert first['candidate_never_started'] and first['original_code']==1
    for key,path in [('collection','capture-collected/capture-collection.json'),('transfer','capture-transfer.json'),
        ('archive','capture-results.tar.gz'),('state','capture-collected/capture-state.json'),
        ('spec','bundle/spec.json'),('deployment','capture-deployment.json')]:
        assert first[key]==pin(FIRST_PROFILE/path),key
    for name,wanted in read(FIRST_PROFILE/'capture-collected/capture-collection.json')['files'].items():
        assert pin(FIRST_PROFILE/'capture-collected'/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='41f2477c7060c3f543471bebe991c77ea3b853d724708fec098a2cbb505cb0a2'
    source=read(SOURCE/'prepared.json');assert source['passed']
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    originals={};prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','statistics_exact.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json']:copy(folder/name,bundle/'evidence'/label/name)
        prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for label,folder in [('baseline',CURRENT),('models',MODELS)]:
        copy(folder/'payload.json',bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    for source,name in [('bundle/spec.json','spec.json'),('capture-collected/capture-collection.json','collection.json'),
        ('capture-collected/capture-state.json','state.json'),('capture-collected/observer-review.json','observer-review.json'),
        ('audit-correction.json','audit-correction.json')]:
        copy(PROFILE/source,bundle/'evidence/profile'/name)
    for source,name in [('bundle/spec.json','initial-spec.json'),('capture-collected/capture-collection.json','initial-collection.json'),
        ('capture-collected/capture-state.json','initial-state.json')]:
        copy(FIRST_PROFILE/source,bundle/'evidence/profile'/name)
    copy(SOURCE/'prepared.json',bundle/'evidence/build/source-prepared.json')
    for role,original in [('current','selected'),('candidate','candidate')]:
        copy(MODELS/'collected'/(original+'-public-512/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(original+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    copy(TOOLS/'README.md',bundle/'prospective-application.md')
    shutil.copy2(ROOT/'.agent/m70-parakeet-observed-dense-where-20260924.md',bundle/'prospective-plan.md')
    models=read(MODELS/'analysis.json')
    stage=dict(passed=True,identities=dict(current=models['identities']['selected'],candidate=models['identities']['candidate']),
        prerequisites=prerequisites,consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    from checks import prereqs
    prereqs(bundle,stage)
    save(bundle/'stage.json',stage)
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
