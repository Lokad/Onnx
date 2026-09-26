"""Bind the original application gates to current root and its one Pad change."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin, read, save
from consumer_scope import verify_scope, PARENT, TRANSPORT, NAMES

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-pad-current-app-amd-20260926'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS = ROOT/'artifacts/parakeet-pad-current-models-amd-20260926'
CONTRACTS = ROOT/'artifacts/parakeet-pad-current-build-amd-20260926'
QUALIFIED = ROOT/'artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925'
SCREEN = ROOT/'artifacts/parakeet-pad-current-screen-amd-20260926'
MEMORY = ROOT/'artifacts/parakeet-pad-memory-diagnostic-amd-20260926'
PRIOR = dict(baseline=CURRENT, models=MODELS, contracts=CONTRACTS,
             qualified=QUALIFIED, screen=SCREEN, memory=MEMORY)
DIGESTS = dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='194eb9a48b64df22d19b51adfc3ebe3accc56004544620123c128c2a76f066bf',
    contracts='1835c79eda505c056cf796702ca734b6e43c65e066b3e54bb566ac25885c4018',
    qualified='c77ef606c76508144c5656bdbe48aeac8948ce28396c8bee645587d31609c475',
    screen='5b8d7df749d600238dfc6e1c9ead50486b0027754656c57d0c553573ada1b17d',
    memory='51e714628218f2908ac46d0e8336b29003d1f7a266e424bd37ec946fa1c7589b')
DIAGNOSIS = ['tests/parakeet/pad-memory-results/managed-diagnosis-20260926.md',
             'tests/parakeet/pad-memory-results/ort-allocation-20260926.md',
             'tests/parakeet/owned-batch-isolation-profile-results/diagnosis-20260925.md']


def previous_closed():
    verify_scope()
    for name, folder in PRIOR.items():
        proof = read(folder/'closed.json')
        assert proof['passed']
        if name in DIGESTS:
            assert pin(folder/'closed.json')['sha256'] == DIGESTS[name]
        assert pin(folder/'analysis.json') == proof['files']['analysis.json']
        for path, wanted in proof.get('files', {}).items():
            assert pin(folder/path) == wanted, path
    assert not read(SCREEN/'closed.json')['admitted']
    assert not read(MEMORY/'closed.json')['admitted']
    models = read(MODELS/'analysis.json')
    assert models['identities']['selected'] == read(QUALIFIED/'analysis.json')['built']
    assert models['identities']['candidate'] == read(CONTRACTS/'analysis.json')['built']


def prepare():
    assert not BASE.exists()
    previous_closed()
    BASE.mkdir()
    bundle = BASE/'bundle'
    bundle.mkdir()
    originals, prerequisites = {}, {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','prerequisites.py','statistics_exact.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for label, folder in PRIOR.items():
        for name in ['closed.json','analysis.json']:
            copy(folder/name, bundle/'evidence'/label/name)
        prerequisites[label] = dict(closed=pin(folder/'closed.json'), analysis=pin(folder/'analysis.json'))
    for label, folder in [('baseline',CURRENT),('models',MODELS)]:
        copy(folder/'payload.json', bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json', bundle/'evidence'/label/'collection.json')
    for role, source in [('current','selected'),('candidate','candidate')]:
        copy(MODELS/'collected'/(source+'-public-512/output/result.json'), bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(source+'-parakeet.json'), bundle/'evidence'/(role+'-parakeet.json'))
    for name in DIAGNOSIS:
        copy(ROOT/name, bundle/'evidence/diagnosis'/Path(name).name)
    copy(TOOLS/'README.md', bundle/'prospective-application.md')
    shutil.copy2(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    models = read(MODELS/'analysis.json')
    screen = read(SCREEN/'analysis.json')
    stage = dict(passed=True, identities=dict(current=models['identities']['selected'], candidate=models['identities']['candidate']),
        prerequisites=prerequisites, consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        failed_component_controls=[r for r in screen['controls'] if not r['passed']], release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    from checks import prereqs
    prereqs(bundle, stage)
    save(bundle/'stage.json', stage)
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf8'), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    for path in [*[PARENT/name for name in [*NAMES,'prerequisites.py']],
                 TRANSPORT/'run.py',TRANSPORT/'remote_prepare.py']:
        originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'))))


if __name__ == '__main__':
    prepare()
