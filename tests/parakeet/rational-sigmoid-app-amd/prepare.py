"""Bind the original application gates to actual root and one fixed rational sigmoid."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin, read, save
from consumer_scope import verify_scope, PARENT, TRANSPORT, NAMES

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-rational-sigmoid-app-amd-20260927'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS = ROOT/'artifacts/parakeet-rational-sigmoid-models-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-rational-sigmoid-build-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-pad-current-root-amd-20260926'
SCREEN = ROOT/'artifacts/parakeet-rational-sigmoid-screen-amd-20260927'
MEMORY = ROOT/'artifacts/parakeet-rational-sigmoid-fallback-diagnostic-amd-20260927'
PRIOR = dict(baseline=CURRENT, models=MODELS, contracts=CONTRACTS,
             qualified=QUALIFIED, screen=SCREEN, memory=MEMORY)
DIGESTS = dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='3d2e01b5aca3f66c434d5d1e8124c2b0d08327bc14dd37409c0e1884619769d8',
    contracts='3301ae58b42e1fd51435f54191cf4bba42c6cbf8e2ad6bf4279af67ae5d3d89a',
    qualified='71c80efd687562cba2e2b5d03e9e036b93de09d0a30f74b976b86355ed12fdf0',
    screen='fc8a6d9736ff3fd324ebd34ad17a213c857bb178556d4aa7175ee455cc8070b6',
    memory='c6f0a34eb3467c69829ad958194f5f88375b83abb12fad8450c233bc41586049')
DIAGNOSIS = ['tests/parakeet/rational-sigmoid-results/models-20260927.md',
             'tests/parakeet/rational-sigmoid-results/fallback-diagnosis-20260927.md',
             'tests/parakeet/rational-sigmoid-results/screen-20260927.md',
             'tests/parakeet/rational-sigmoid-results/contracts-20260927.md',
             'tests/parakeet/ort-activation-review/diagnosis-20260926.md',
             'tests/parakeet/pad-current-profile-results/diagnosis-20260927.md']


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
    assert models['identities']['candidate'] == read(CONTRACTS/'analysis.json')['product']


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
