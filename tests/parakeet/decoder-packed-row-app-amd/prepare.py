"""Freeze the independent complete application decision for one unchanged candidate."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin, read, save
from consumer_scope import verify_scope, PARENT, TRANSPORT, NAMES, PREVIOUS

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-app-amd-20260927'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS = ROOT/'artifacts/parakeet-decoder-packed-row-models-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v3-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
SCREEN = ROOT/'artifacts/parakeet-decoder-packed-row-screen-v2-amd-20260927'
DIAGNOSTIC = ROOT/'artifacts/parakeet-decoder-unmapped-calls-amd-20260927'
PRIOR = dict(baseline=CURRENT, models=MODELS, contracts=CONTRACTS,
             qualified=QUALIFIED, screen=SCREEN, diagnosis=DIAGNOSTIC)
DIGESTS = dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='5d6832083d103bef9db7bb733b1e96decbae22625f3138680ab6f9a426c78e3b',
    contracts='fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d',
    qualified='f6df53ce2ab773898bc84f144abeda97908d383fef9a4df4b7299a22a4d3594d',
    screen='3e6a7562db1946954f2cddba10b2ee31958854e66c50fbb55848ed04e3e75148',
    diagnosis='966941d1e040211b840259438fe8bb786aea23dc3c8f9987208cf242b5a7e986')
DIAGNOSIS = ['tests/parakeet/decoder-packed-row-results/screen-20260927.md',
             'tests/parakeet/decoder-packed-row-results/unmapped-calls-20260927.md',
             'tests/parakeet/decoder-packed-row-results/README.md',
             'tests/parakeet/decoder-current-review/next-observation-20260927.md']


def previous_closed():
    verify_scope()
    assert all(DIGESTS.values()), 'Bind the completed model audit before preparation'
    for name, folder in PRIOR.items():
        proof = read(folder/'closed.json')
        assert proof['passed'] and pin(folder/'closed.json')['sha256'] == DIGESTS[name]
        assert pin(folder/'analysis.json') == proof['files']['analysis.json']
        for path, wanted in proof['files'].items(): assert pin(folder/path) == wanted, path
    assert not read(SCREEN/'closed.json')['admitted'] and not read(DIAGNOSTIC/'closed.json')['admitted']
    models = read(MODELS/'analysis.json')
    assert models['identities']['selected'] == read(QUALIFIED/'analysis.json')['built']
    assert models['identities']['candidate']['Lokad.Onnx.dll'] == read(CONTRACTS/'analysis.json')['products']['candidate']
    applied = read(QUALIFIED/'bundle/evidence/root-applied.json')
    for name, wanted in applied['source_files'].items(): assert pin(ROOT/name) == wanted, name


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
    for path in [*[PARENT/name for name in [*NAMES,'checks.py','test_admission.py','prerequisites.py']], PREVIOUS/'remote_prepare.py',
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
