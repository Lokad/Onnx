"""Bind the exact qualified eight-case graph protocol to the single collapsed-axis transpose pair."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile

PARENT = Path(__file__).resolve().parent.parent/'pad-current-graphs-amd'
sys.path.insert(1, str(PARENT))
from protocol import CASES, JOBS, LIMITS, pin, read, save
from consumer_scope import verify_scope
from prerequisites import validate

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-transpose-axis-graphs-amd-20260928'
GRAPH = ROOT/'artifacts/parakeet-attention-owned-graphs-amd-20260928'
PRODUCT = ROOT/'artifacts/parakeet-transpose-axis-models-amd-20260928'
PRIOR = dict(graph=GRAPH, models=PRODUCT,
    app=ROOT/'artifacts/parakeet-transpose-axis-app-amd-20260928',
    shared=ROOT/'artifacts/parakeet-transpose-axis-shared-amd-20260928',
    pyannote=ROOT/'artifacts/parakeet-transpose-axis-pyannote-amd-20260928')
REMOTE = dict(graph='/dev/shm/lokad-attention-owned-graphs-20260928',
    models='/dev/shm/lokad-transpose-axis-models-20260928',
    app='/dev/shm/lokad-transpose-axis-app-20260928',
    shared='/dev/shm/lokad-transpose-axis-shared-20260928',
    pyannote='/dev/shm/lokad-transpose-axis-pyannote-20260928')
DIGESTS = dict(graph='0878b709926c77a55eaf82e63194e6d1379e1c877cfd11b7c0ab997e142eb613',
    models='8568a2299a820273b7357eaaba9c151bc83121bfc781759afbe9142b8d7b5984',
    app='2e4741a67772d71e4442630d7f9cf751dfba2a52e12631fffed3b2db62fefd85',
    shared='fa74521cbe415ead594903a2b76649bcd4ec2edc20d075f616ae67948e920f1b',
    pyannote='3188186fc659d1846fccb3b6c251f7a91f24ecda641dff32ce052b98f1c972ca')
SCOPE = ROOT/'tests/parakeet/rational-sigmoid-results/graph-scope-20260927.json'


def previous_closed():
    verify_scope(); assert all(DIGESTS.values()), 'Bind actual application/shared/Pyannote closures first'
    reports = {}
    for label, folder in PRIOR.items():
        assert pin(folder/'closed.json')['sha256'] == DIGESTS[label]
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        reports[label] = read(folder/'analysis.json')
    assert read(PRIOR['app']/'closed.json')['admitted'] and read(GRAPH/'closed.json')['admitted']
    compatible = read(PRODUCT/'collected/evidence/compatibility.json')
    products = validate(reports, compatible, read(SCOPE))
    for row in read(SCOPE)['models']:
        assert pin(ROOT/row['model']) == row['identity']
    return products


def prepare():
    assert not BASE.exists(); products = previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = verify_scope()
    def copy(source, target):
        target = bundle/target; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'checks_e5.py', 'short_checks.py',
                 'reuse.py', 'native.py', 'native-e5.py', 'native-short.py']:
        copy((TOOLS if name == 'remote_prepare.py' else PARENT)/name, 'tools/'+name)
    copy(TOOLS/'README.md', 'README.md'); shutil.copy2(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    for label, folder in PRIOR.items():
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder/name, 'evidence/prerequisites/'+label+'/'+name)
        copy(folder/'collected/collection.json', 'evidence/prerequisites/'+label+'/collection.json')
    copy(PRODUCT/'collected/evidence/compatibility.json', 'evidence/product-compatibility.json')
    copy(SCOPE, 'evidence/graph-scope.json'); copy(GRAPH/'collected/cases.json', 'cases.json')
    graph = read(GRAPH/'payload.json'); links = {}; collection = read(GRAPH/'collected/collection.json')
    for name, wanted in graph['files'].items():
        if name.startswith(('reference/', 'runtimes/', 'runtimes-e5/', 'runtimes-short/',
                            'evidence/warmed-consumer/', 'evidence/e5-consumer/', 'evidence/short-consumer/')) or name in ['evidence/baseline/payload.json', 'source/global.json']:
            assert pin(GRAPH/'collected'/name) == wanted == collection['files'][name]
            links[name] = dict(source=REMOTE['graph']+'/'+name, identity=wanted)
    assert 'source/global.json' in links
    for prefix in ['runtimes', 'runtimes-e5', 'runtimes-short']:
        for role, label in [('current', 'selected'), ('candidate', 'candidate')]:
            name = prefix+'/'+role+'/Lokad.Onnx.dll'; source = PRODUCT/'collected/runtimes'/label/'Lokad.Onnx.dll'
            assert pin(source) == products[role]['Lokad.Onnx.dll']
            links[name] = dict(source=REMOTE['models']+'/runtimes/'+label+'/Lokad.Onnx.dll', identity=pin(source))
    old_built = read(GRAPH/'collected/built.json')
    built = dict(old_built, files={name: row['identity'] for name, row in links.items() if Path(name).name.startswith('ReleaseBenchmark.')})
    save(bundle/'built.json', built)
    stage = dict(passed=True, links=links, products=products, prerequisites=REMOTE,
        previous_owner=read(PRIOR['pyannote']/'collected/collection.json')['identities'][0],
        **{key: built[key] for key in ['consumer', 'e5_consumer', 'short_consumer']},
        **{key: graph[key] for key in ['previous_consumer', 'external', 'interpreter', 'python_paths']},
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    for path in [*TOOLS.iterdir(), *PARENT.iterdir()]:
        if path.is_file():
            if path.suffix == '.py': ast.parse(path.read_text(encoding='utf8'), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'), jobs=len(JOBS), limits=LIMITS)))


if __name__ == '__main__': prepare()
