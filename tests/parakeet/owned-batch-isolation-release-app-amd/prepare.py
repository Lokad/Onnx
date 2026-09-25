"""Bind the unchanged full-request comparison to the actual released product."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin, read, save
from consumer_scope import verify_scope, expected, NAMES
from release_equality import BASE as EQUALITY, verify as verify_equality

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-release-app-amd-20260925'
CURRENT = ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS = ROOT/'artifacts/parakeet-owned-batch-isolation-models-amd-20260925'
RELEASE_MODELS = ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
CONTRACTS = ROOT/'artifacts/parakeet-owned-batch-isolation-build-amd-20260925'
PARENT_APP = ROOT/'artifacts/parakeet-owned-batch-isolation-parent-app-amd-20260925'
GRAPHS = ROOT/'artifacts/parakeet-owned-batch-graph-qualification-20260925'
PRIOR = dict(baseline=CURRENT, models=MODELS, release_models=RELEASE_MODELS,
    contracts=CONTRACTS, parent_app=PARENT_APP, graphs=GRAPHS, equality=EQUALITY)
DIGESTS = dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='1997eeb782df89975f4893820bc0fc246dec60d1839b2c63a86ad2bba1aecd9d',
    release_models='5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842',
    contracts='5dd53e90d4924a36bb6f43cc80fa939e9236492949542dcf659dc056d4ea3860',
    graphs='ec95b9c7f8019b402fe513b6da6938cf83a8bc2c91f1d5ab65fedd4f7b55ed7c')


def previous_closed():
    # The parent comparison must close successfully before preparation/staging.
    parent = read(PARENT_APP/'closed.json')
    assert parent['passed'] and parent['admitted']
    verify_scope()
    verify_equality()
    for name, folder in PRIOR.items():
        proof = read(folder/'closed.json')
        assert proof['passed']
        if name in DIGESTS:
            assert pin(folder/'closed.json')['sha256'] == DIGESTS[name]
        recorded = proof['files']['analysis.json'] if name == 'graphs' else proof['analysis']
        assert recorded == pin(folder/'analysis.json')
        root = ROOT if proof.get('paths_relative_to_repository') else folder
        for path, wanted in proof.get('files', {}).items():
            assert pin(root/path) == wanted, path
    graph = read(GRAPHS/'closed.json')
    assert graph['admitted'] and graph['all_controls_passed']


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

    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'prerequisites.py', 'statistics_exact.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for label, folder in PRIOR.items():
        for name in ['closed.json', 'analysis.json']:
            copy(folder/name, bundle/'evidence'/label/name)
        prerequisites[label] = dict(closed=pin(folder/'closed.json'), analysis=pin(folder/'analysis.json'))
    for label, folder in [('baseline', CURRENT), ('models', MODELS), ('release_models', RELEASE_MODELS)]:
        copy(folder/'payload.json', bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json', bundle/'evidence'/label/'collection.json')
    copy(PARENT_APP/'collected/collection.json', bundle/'evidence/parent_app/collection.json')
    copy(CONTRACTS/'build-review.json', bundle/'evidence/contracts/build-review.json')
    for role, folder, source in [('current', RELEASE_MODELS, 'selected'), ('candidate', MODELS, 'candidate')]:
        copy(folder/'collected'/(source+'-public-512/output/result.json'), bundle/'evidence'/(role+'-public.json'))
        copy(folder/'collected/manifests'/(source+'-parakeet.json'), bundle/'evidence'/(role+'-parakeet.json'))
    copy(TOOLS/'README.md', bundle/'prospective-application.md')
    shutil.copy2(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    models, release = read(MODELS/'analysis.json'), read(RELEASE_MODELS/'analysis.json')
    stage = dict(passed=True, identities=dict(current=release['identities']['selected'], candidate=models['identities']['candidate']),
        prerequisites=prerequisites, consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        failed_graph_cases=[], release_admitted=False,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    from checks import prereqs
    prereqs(bundle, stage)
    save(bundle/'stage.json', stage)
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf8'), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    for name in NAMES:
        source, _ = expected(name)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz', dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'))))


if __name__ == '__main__':
    prepare()
