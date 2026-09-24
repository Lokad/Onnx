"""Freeze the unchanged application comparison after full numerical qualification."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-app-amd-20260924'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS=ROOT/'artifacts/parakeet-prepared-recurrence-models-amd-20260924'
BUILD=ROOT/'artifacts/parakeet-prepared-recurrence-build-review-20260924'
RAW_BUILD=ROOT/'artifacts/parakeet-prepared-recurrence-build-amd-20260924'
CONTRACTS=ROOT/'artifacts/parakeet-prepared-recurrence-contracts-amd-20260924'
RESIDENCY=ROOT/'artifacts/parakeet-prepared-recurrence-calls-amd-v2-20260924'
COMPONENT=ROOT/'artifacts/parakeet-prepared-recurrence-timing-amd-20260924'
QUALIFIED=ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
PRIOR={'baseline':(CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'),
 'models':(MODELS,'c0b2f61442420eed2b9afcc94263eda9917edb6de84f09ec2016d3a850077d9d'),
 'build':(BUILD,'1b7f8e2130d490851e861790b2e051edfca3308be6d6d9e1067cda8b97f9869f'),
 'contracts':(CONTRACTS,'aacf2dbe9265a96c5699155a86b1743462387990245ca8fdc6cc0e75cc73952a'),
 'residency':(RESIDENCY,'20b3bc4b5a2d2ddb6e0834e40fb7e2bcb5220dd1f45c813a7a830466ddbc463d'),
 'component':(COMPONENT,'d0eb4898dbc850d2b29659bd47b71181262ae4923c88760fa7f6995348b494b2'),
 'root':(QUALIFIED,'16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73')}


def previous_closed():
    from consumer_scope import verify_scope
    assert verify_scope()
    for folder,digest in PRIOR.values():
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        root=ROOT if proof.get('paths_relative_to_repository') else folder
        for name,wanted in proof['files'].items():assert pin(root/name)==wanted,name
    raw=read(RAW_BUILD/'closed.json')
    assert pin(RAW_BUILD/'closed.json')==read(BUILD/'closed.json')['original_refusal'] and not raw['passed'] and raw['build_jobs_passed']
    for name,wanted in raw['files'].items():assert pin(RAW_BUILD/name)==wanted,name
    assert read(CONTRACTS/'analysis.json')['candidate_passes']==300
    assert read(CURRENT/'analysis.json')['performance']['baseline_valid']
    component=read(COMPONENT/'analysis.json')['performance']
    assert not component['admitted'] and not component['controls_passed'] and sum(not r['passed'] for r in component['controls'])==40
    source_path=ROOT/'artifacts/parakeet-prepared-recurrence-source-20260924/prepared.json'
    assert pin(source_path)['sha256']=='b52a89c1043165de1c376b37fc5307cd003a7c8b76f0f52508c4cdefcc669eab'
    for name,wanted in read(source_path)['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={};prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','statistics_exact.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for label,(folder,_) in PRIOR.items():
        for name in ['closed.json','analysis.json']:copy(folder/name,bundle/'evidence'/label/name)
        transport=RAW_BUILD if label=='build' else folder
        copy(transport/'payload.json',bundle/'evidence'/label/'payload.json')
        copy(transport/'collected/collection.json',bundle/'evidence'/label/'collection.json')
        prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    copy(RAW_BUILD/'closed.json',bundle/'evidence/build/original-refusal.json')
    for role,original in [('current','selected'),('candidate','candidate')]:
        copy(MODELS/'collected'/(original+'-public-512/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(original+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    copy(TOOLS/'README.md',bundle/'prospective-application.md')
    shutil.copy2(ROOT/'.agent/m64-parakeet-prepared-recurrence-20260924.md',bundle/'prospective-plan.md')
    models=read(MODELS/'analysis.json')
    stage=dict(passed=True,identities=dict(current=models['identities']['selected'],candidate=models['identities']['candidate']),
        prerequisites=prerequisites,consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
