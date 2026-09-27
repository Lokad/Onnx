"""Freeze the original application verdict for the diagnosed LSTM layout."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin, read, save
from consumer_scope import verify_scope, PARENT, TRANSPORT, NAMES

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-app-amd-20260927'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS = ROOT/'artifacts/parakeet-decoder-lstm-layout-models-amd-20260927'
CONTROL = ROOT/'artifacts/parakeet-decoder-lstm-layout-scalar-baseline-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
SCREEN = ROOT/'artifacts/parakeet-decoder-lstm-layout-timing-amd-20260927'
DIAGNOSTIC = ROOT/'artifacts/parakeet-decoder-lstm-runtime-analysis-20260927'
PRIOR = dict(baseline=CURRENT, models=MODELS, control=CONTROL,
             qualified=QUALIFIED, screen=SCREEN, diagnosis=DIAGNOSTIC)
DIGESTS = dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='1acb1d884eb5293e0487e1b34d8c842b7240823040e63d6dc2a043ec89f7a111',
    control='63c97822999a74921c2e8a3c64e0af9ec6682a6829ba83de52bba353b93b85ec',
    qualified='d0a78cdd3106d6a72a41303f879bbb2f9ea3bd6a298d778333015f38ccdac246',
    screen='ac1b5e0e5400d03d84250d6148253208a6d25d3bc9e565a49718721861772fe2',
    diagnosis='dfa4cef574312c04e8d86e6c69151133b2ab4b4f8646bd74fcbe77c12ef23c68')
DIAGNOSIS = ['tests/parakeet/decoder-lstm-layout-results/timing-20260927.md',
             'tests/parakeet/decoder-lstm-layout-results/contracts-20260927.md',
             'tests/parakeet/decoder-lstm-runtime-results/README.md',
             'tests/parakeet/decoder-lstm-runtime-results/runtime-20260927.json',
             'tests/parakeet/decoder-lstm-current-review/next-decision-20260927.md']


def previous_closed():
    verify_scope()
    assert all(DIGESTS.values()), 'Bind the successful complete-model audit before preparation'
    for name, folder in PRIOR.items():
        proof = read(folder/'closed.json')
        assert proof['passed'] and pin(folder/'closed.json')['sha256'] == DIGESTS[name]
        assert pin(folder/'analysis.json') == proof['files']['analysis.json']
        for path, wanted in proof['files'].items(): assert pin(folder/path) == wanted, path
    assert not read(SCREEN/'closed.json')['admitted']
    models = read(MODELS/'analysis.json'); control = read(CONTROL/'analysis.json')
    assert models['identities'] == dict(selected=control['baseline'], candidate=control['candidate'])
    assert models['identities']['selected'] == read(QUALIFIED/'analysis.json')['built']
    applied = read(QUALIFIED/'bundle/evidence/root-applied.json')
    for name, wanted in applied['source_files'].items(): assert pin(ROOT/name) == wanted, name


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals, prerequisites = {}, {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target); originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'prerequisites.py', 'statistics_exact.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for label, folder in PRIOR.items():
        for name in ['closed.json', 'analysis.json']: copy(folder/name, bundle/'evidence'/label/name)
        prerequisites[label] = dict(closed=pin(folder/'closed.json'), analysis=pin(folder/'analysis.json'))
    for label, folder in [('baseline', CURRENT), ('models', MODELS)]:
        copy(folder/'payload.json', bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json', bundle/'evidence'/label/'collection.json')
    for role, source in [('current', 'selected'), ('candidate', 'candidate')]:
        copy(MODELS/'collected'/(source+'-public-512/output/result.json'), bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(source+'-parakeet.json'), bundle/'evidence'/(role+'-parakeet.json'))
    for name in DIAGNOSIS: copy(ROOT/name, bundle/'evidence/diagnosis'/Path(name).name)
    copy(TOOLS/'README.md', bundle/'prospective-application.md')
    shutil.copy2(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    models = read(MODELS/'analysis.json'); screen = read(SCREEN/'analysis.json')['performance']
    stage = dict(passed=True, identities=dict(current=models['identities']['selected'], candidate=models['identities']['candidate']),
        prerequisites=prerequisites, consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        failed_component_controls=[r for r in screen['controls'] if not r['passed']], release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    from checks import prereqs
    prereqs(bundle, stage); save(bundle/'stage.json', stage)
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix == '.py': ast.parse(path.read_text(encoding='utf8'), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    for path in [*[PARENT/name for name in [*NAMES, 'prerequisites.py', 'remote_prepare.py']], TRANSPORT/'run.py']:
        originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz', dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'))))


if __name__ == '__main__': prepare()
