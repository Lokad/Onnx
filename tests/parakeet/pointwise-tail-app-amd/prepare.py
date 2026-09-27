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
BASE = ROOT/'artifacts/parakeet-pointwise-tail-app-amd-20260927'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS = ROOT/'artifacts/parakeet-pointwise-tail-models-amd-20260927'
CONTROL = ROOT/'artifacts/parakeet-pointwise-tail-arithmetic-contracts-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-lstm-layout-root-amd-20260927'
SCREEN = ROOT/'artifacts/parakeet-pointwise-tail-timing-amd-20260927'
DIAGNOSTIC = ROOT/'artifacts/parakeet-pointwise-tail-runtime-analysis-20260927'
PRIOR = dict(baseline=CURRENT,models=MODELS,control=CONTROL,qualified=QUALIFIED,screen=SCREEN,diagnosis=DIAGNOSTIC)
DIGESTS = dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='f773277daa848121c199c58e67bce3777677c2ee2654fbe26250d02719155960',
    control='558f2a523febd0d794bc7da6cdae3381513b7ff3d75dfd4ca4f133f618821cc8',
    qualified='efb99eea455647c64bbe0811c1b7d46387add59049ac81e48a926b250e7c42da',
    screen='b9ff0d6050068ad8ad9ed9fa7a50350bb7e99079e10316f077183bea8ef0765c',
    diagnosis='75f3b25985a8ff4efdc7465de0444b3e667deb0c5922755b0da8a8a5770601c4')
DIAGNOSIS = [f'tests/parakeet/decoder-lstm-layout-profile-results/pointwise-tail-{name}-20260927.md'
    for name in ['nan','timing','runtime']]


def previous_closed():
    verify_scope(); assert all(DIGESTS.values()),'Bind successful complete-model audit before preparation'
    for name,folder in PRIOR.items():
        proof = read(folder/'closed.json'); assert pin(folder/'closed.json')['sha256'] == DIGESTS[name]
        if name == 'control': assert proof['completed'] and proof['arithmetic_contract_passed']
        elif name == 'screen': assert proof['completed'] and not proof['component_admitted']
        else: assert proof['passed']
        assert pin(folder/'analysis.json') == proof['files']['analysis.json']
        for path,wanted in proof['files'].items(): assert pin(folder/path) == wanted,path
    models = read(MODELS/'analysis.json'); control = read(CONTROL/'analysis.json')
    assert models['identities'] == dict(selected=control['products']['baseline'],candidate=control['products']['candidate'])
    assert models['identities']['selected'] == read(QUALIFIED/'analysis.json')['built']
    source = {n.removeprefix('source/'):w for n,w in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(source) == 443
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
    models = read(MODELS/'analysis.json'); screen = read(SCREEN/'analysis.json')['performance']
    stage = dict(passed=True,identities=dict(current=models['identities']['selected'],candidate=models['identities']['candidate']),
        prerequisites=prerequisites,consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        failed_component_controls=[r for r in screen['controls'] if not r['passed']],release_admitted=False,
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
