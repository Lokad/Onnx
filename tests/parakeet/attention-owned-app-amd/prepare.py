"""Freeze one unchanged-candidate application verdict after model correctness."""
import ast
import json
import shutil
import sys
import tarfile
from pathlib import Path
from consumer_scope import verify_scope,PARENT,TRANSPORT,NAMES,remote_preparation

sys.path.insert(1,str(PARENT))
from protocol import pin,read,save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-attention-owned-app-amd-20260928'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS = ROOT/'artifacts/parakeet-attention-owned-models-amd-20260928'
CONTROL = ROOT/'artifacts/parakeet-attention-owned-build-amd-20260928'
QUALIFIED = ROOT/'artifacts/parakeet-pointwise-tail-root-amd-20260927'
CENSUS = ROOT/'artifacts/parakeet-attention-owned-census-amd-20260928'
PRIOR = dict(baseline=CURRENT,models=MODELS,control=CONTROL,qualified=QUALIFIED,census=CENSUS)
DIGESTS = dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='2d64fad91c26c00ffe7cd003aa128efc4bc92280da96a9bb805a8e2b93412177',
    control='a293403a5dc3a7bdabb7fdc448470feea2b54dbf139d98b7b3f9ed0eef39854f',
    qualified='fc11676361a50ef613f783fcb51e4ead488c9c2ee34a3edddcbb561f67037e47',
    census='ef4e41eca49d33f6cac4ae7d510922053874002787709cc3d83bc98c3437d3d7')
DIAGNOSIS = ['tests/parakeet/attention-cost-results/results-20260927.md',
    'tests/parakeet/pointwise-tail-profile-results/attention-routes-20260927.md']


def previous_closed():
    verify_scope(); assert all(DIGESTS.values()),'Bind successful complete-model audit before preparation'
    for name,folder in PRIOR.items():
        proof = read(folder/'closed.json'); assert pin(folder/'closed.json')['sha256'] == DIGESTS[name]
        assert proof['passed']
        assert pin(folder/'analysis.json') == proof['files']['analysis.json']
        for path,wanted in proof['files'].items(): assert pin(folder/path) == wanted,path
    models = read(MODELS/'analysis.json'); control = read(CONTROL/'analysis.json')
    assert models['identities']['candidate'] == control['product']
    assert read(CENSUS/'analysis.json')['product'] == control['product']
    assert models['identities']['selected'] == read(QUALIFIED/'analysis.json')['built']
    source = {n.removeprefix('source/'):w for n,w in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(source) == 445
    for name,wanted in source.items(): assert pin(ROOT/name) == wanted,name


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals,prerequisites = {},{}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target); originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['protocol.py','remote.py','checks.py','statistics_exact.py']: copy(PARENT/name,bundle/'tools'/name)
    copy(TOOLS/'prerequisites.py',bundle/'tools/prerequisites.py')
    (bundle/'tools/remote_prepare.py').write_text(remote_preparation(),encoding='utf8')
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json']: copy(folder/name,bundle/'evidence'/label/name)
        prerequisites[label] = dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for label,folder in [('baseline',CURRENT),('models',MODELS)]:
        copy(folder/'payload.json',bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    copy(MODELS/'bundle/evidence/compatibility.json',bundle/'evidence/models-compatibility.json')
    for role,source in [('current','selected'),('candidate','candidate')]:
        copy(MODELS/'collected'/(source+'-public-512/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(source+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    for name in DIAGNOSIS: copy(ROOT/name,bundle/'evidence/diagnosis'/Path(name).name)
    copy(TOOLS/'README.md',bundle/'prospective-application.md')
    shutil.copy2(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    models = read(MODELS/'analysis.json')
    stage = dict(passed=True,identities=dict(current=models['identities']['selected'],candidate=models['identities']['candidate']),
        prerequisites=prerequisites,consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        failed_component_controls=[],release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    from checks import prereqs
    assert prereqs(bundle,stage)['passed']; save(bundle/'stage.json',stage)
    for path in [*TOOLS.iterdir(),*[PARENT/name for name in [*NAMES,'prerequisites.py','remote_prepare.py']],TRANSPORT/'run.py']:
        if path.is_file():
            if path.suffix == '.py': ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    ast.parse(remote_preparation(),'remote_prepare.py')
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__ == '__main__': prepare()
