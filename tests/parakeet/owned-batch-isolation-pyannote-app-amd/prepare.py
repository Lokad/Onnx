"""Freeze the existing Pyannote application and meeting checks for exact products."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from checks import prereqs
from consumer_scope import verify_scope

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-pyannote-app-amd-20260925'
OLD = ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924'
APP_PAYLOAD = OLD/'collected'
GRAPHS = ROOT/'artifacts/parakeet-owned-batch-graph-qualification-20260925'
PRIOR = dict(models=ROOT/'artifacts/parakeet-owned-batch-isolation-pyannote-amd-20260925',
    parakeet=ROOT/'artifacts/parakeet-owned-batch-isolation-models-amd-20260925',
    shared=ROOT/'artifacts/parakeet-owned-batch-isolation-shared-amd-20260925',
    equality=ROOT/'artifacts/parakeet-owned-batch-release-equality-20260925', baseline=OLD)
PRIOR['parakeet-release'] = ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
PRIOR['parakeet-app'] = ROOT/'artifacts/parakeet-owned-batch-isolation-release-app-amd-20260925'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
GRAPH_HELPER = ROOT/'tests/benchmarks/e5-steady-short-results/qualified_graphs.py'
EQUALITY_HELPER = ROOT/'tests/parakeet/owned-batch-isolation-release-app-amd/release_equality.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


monitor = load('application_monitor', MONITOR)


def previous_closed():
    assert read(PRIOR['models']/'closed.json')['passed']
    assert read(PRIOR['parakeet-app']/'closed.json')['admitted']
    verify_scope()
    assert pin(OLD/'closed.json')['sha256'] == '5dfd12f296a49b40e8731d77289fc94198df9ab2fc64a7e0252d7b3d8a2af4b0'
    assert pin(PRIOR['parakeet']/'closed.json')['sha256'] == '1997eeb782df89975f4893820bc0fc246dec60d1839b2c63a86ad2bba1aecd9d'
    load('retained_graph_qualification', GRAPH_HELPER).admission()
    load('retained_parakeet_equality', EQUALITY_HELPER).verify()
    for folder in [*PRIOR.values(), GRAPHS]:
        proof = read(folder/'closed.json')
        assert proof['passed']
        for name, wanted in proof['files'].items():
            assert pin(folder/name) == wanted, name
    identities = read(PRIOR['models']/'analysis.json')['identities']
    assert read(PRIOR['shared']/'analysis.json')['identities'] == identities
    assert read(PRIOR['parakeet-app']/'analysis.json')['identities'] == dict(current=identities['selected'], candidate=identities['candidate'])


def prepare():
    assert not BASE.exists()
    previous_closed()
    BASE.mkdir()
    bundle = BASE/'bundle'
    bundle.mkdir()
    originals = verify_scope()
    prerequisites = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'prerequisites.py',
                 'graph_prerequisite.py', 'meeting_protocol.py', 'meetings_audit.py', 'admission.py', 'semantics.py']:
        copy(TOOLS/name, bundle/'tools'/name)
    for name in ['closed.json', 'analysis.json']:
        copy(GRAPHS/name, bundle/'evidence/graph-qualification'/name)
    for label, folder in PRIOR.items():
        for name in ['closed.json', 'analysis.json']:
            copy(folder/name, bundle/'evidence'/label/name)
        if label != 'equality':
            copy(folder/'payload.json', bundle/'evidence'/label/'payload.json')
            copy(folder/'collected/collection.json', bundle/'evidence'/label/'collection.json')
        prerequisites[label] = dict(closed=pin(folder/'closed.json'), analysis=pin(folder/'analysis.json'))
    for family in ['pyannote', 'parakeet']:
        copy(APP_PAYLOAD/'manifests'/('candidate-'+family+'.json'), bundle/'evidence'/('original-'+family+'.json'))
    copy(APP_PAYLOAD/'meetings/manifest.json', bundle/'evidence/original-meetings.json')
    copy(APP_PAYLOAD/'meetings-run/output/result.json', bundle/'evidence/selected-meetings.json')
    copy(TOOLS/'README.md', bundle/'prospective-plan.md')
    stage = dict(passed=True, identities=read(PRIOR['models']/'analysis.json')['identities'], prerequisites=prerequisites,
        graph_qualification=dict(closed=pin(GRAPHS/'closed.json'), analysis=pin(GRAPHS/'analysis.json')),
        consumers=read(OLD/'analysis.json')['consumers'],
        files={path.relative_to(bundle).as_posix(): pin(path) for path in bundle.rglob('*') if path.is_file()})
    prereqs(bundle, stage)
    save(bundle/'stage.json', stage)
    for path in [*TOOLS.iterdir(), MONITOR, GRAPH_HELPER, EQUALITY_HELPER]:
        if path.is_file():
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf8'), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'))))


if __name__ == '__main__':
    prepare()
