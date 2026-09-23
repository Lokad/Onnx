"""Freeze a six-process Parakeet comparison only after exact model qualification."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-wide-entry-first-use-app-amd-v2-20260923'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS=ROOT/'artifacts/parakeet-wide-entry-first-use-models-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
SCREEN=ROOT/'artifacts/parakeet-wide-entry-first-use-screen-amd-20260923'
QUALIFIED=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
PRIOR={'baseline':(CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'),
 'models':(MODELS,'f30100534cbb79db790aac30365d533e6d3dcce79e779abc5feb8f7cf3fc1e22'),
 'build':(BUILD,'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
 'screen':(SCREEN,'5ca99bc1ac4b647c083104fc7f72a3cd19871c5c72499ff085c9475ca4b8cb8a'),
 'isolated-build':(ROOT/'artifacts/parakeet-isolated-short-kernels-build-amd-v2-20260923','0adb72e2eae376df7d4f3dd6eb7f4a97b15dd57c5d4c3fa962f8a4cd3a9c45c6'),
 'wide-build':(ROOT/'artifacts/parakeet-wide-projection-isolation-build-amd-20260923','c67b21b2e3f9d3d09f822ab075e649e232c6c42d7e7503470711d21a1bf5604d'),
 'root':(QUALIFIED,'62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0')}

def previous_closed():
    from consumer_scope import verify_scope
    assert verify_scope()
    for folder,digest in PRIOR.values():
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(SCREEN/'analysis.json')['admitted'] and read(CURRENT/'analysis.json')['performance']['baseline_valid']
    source=read(ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923/prepared.json')
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
        copy(MODELS/'collected'/(original+'-public/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(original+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    shutil.copy2(ROOT/'.agent/m54-wide-projection-first-use-entry-20260923.md',bundle/'prospective-plan.md')
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
