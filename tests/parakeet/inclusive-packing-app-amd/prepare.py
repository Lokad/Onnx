"""Freeze a six-process Parakeet comparison only after exact model qualification."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-inclusive-packing-app-amd-20260924'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS=ROOT/'artifacts/parakeet-inclusive-packing-models-amd-20260924'
BUILD=ROOT/'artifacts/parakeet-inclusive-packing-build-amd-20260924'
CONTRACTS=ROOT/'artifacts/parakeet-inclusive-packing-contracts-amd-v2-20260924'
RESIDENCY=ROOT/'artifacts/parakeet-inclusive-packing-residency-amd-v2-20260924'
QUALIFIED=ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
PRIOR={'baseline':(CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'),
 'models':(MODELS,'2d6aedeeaa8cdec4083a462f117272881c544f79f1464c6425e451f53fbfc66c'),
 'build':(BUILD,'50f2a3a8a2ebbe20d41315bc0be4242e24fb7f3eb74e78533091a750791a4078'),
 'contracts':(CONTRACTS,'e85e5f5f7d435141b41b19957cc9bc119850c0593a120956002c798e66d01b76'),
 'residency':(RESIDENCY,'82342a58fddab63974e8bff009916a91376beb1d91f9db264480c2763c8e7b0a'),
 'root':(QUALIFIED,'16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73')}

def previous_closed():
    from consumer_scope import verify_scope
    assert verify_scope()
    for folder,digest in PRIOR.values():
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(RESIDENCY/'analysis.json')['comparison']['both_modes_exact']
    assert read(CONTRACTS/'analysis.json')['candidate_passes']==124
    assert read(CURRENT/'analysis.json')['performance']['baseline_valid']
    source_path=ROOT/'artifacts/parakeet-inclusive-packing-source-20260924/prepared.json'
    assert pin(source_path)['sha256']=='a535266a48631edc8578eeab679df644884d08f1f1cb08e84dbb27415b1cbd7c'
    source=read(source_path)
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={};prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','statistics_exact.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for label,(folder,_) in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
        prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for role,original in [('current','selected'),('candidate','candidate')]:
        copy(MODELS/'collected'/(original+'-public-512/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(original+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    shutil.copy2(ROOT/'.agent/m63-parakeet-inclusive-packing-20260924.md',bundle/'prospective-plan.md')
    models=read(MODELS/'analysis.json')
    stage=dict(passed=True,identities=dict(current=models['identities']['selected'],candidate=models['identities']['candidate']),
        prerequisites=prerequisites,consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))

if __name__=='__main__':prepare()
