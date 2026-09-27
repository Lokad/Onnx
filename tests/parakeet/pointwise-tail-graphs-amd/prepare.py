"""Bind the exact qualified eight-case graph protocol to the fixed pointwise remainder pair."""
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
BASE = ROOT/'artifacts/parakeet-pointwise-tail-graphs-amd-20260927'
GRAPH = ROOT/'artifacts/parakeet-decoder-lstm-layout-graphs-amd-20260927'
PRODUCT = ROOT/'artifacts/parakeet-pointwise-tail-models-amd-20260927'
PRIOR = dict(graph=GRAPH, models=PRODUCT,
    app=ROOT/'artifacts/parakeet-pointwise-tail-app-amd-20260927',
    shared=ROOT/'artifacts/parakeet-pointwise-tail-shared-amd-20260927',
    pyannote=ROOT/'artifacts/parakeet-pointwise-tail-pyannote-amd-20260927')
REMOTE = dict(graph='/dev/shm/lokad-lstmlayout-graphs-20260927',
    models='/dev/shm/lokad-pwt-models-20260927',
    app='/dev/shm/lokad-pwt-app-20260927',
    shared='/dev/shm/lokad-pwt-shared-20260927',
    pyannote='/dev/shm/lokad-pwt-pyannote-20260927')
DIGESTS = dict(graph='f7b0a361e4019f44541efe270e80e55b3bbadf341fa97820c88d1f077c9bdf48',
    models='f773277daa848121c199c58e67bce3777677c2ee2654fbe26250d02719155960',
    app='505da6ab8c9ce1d61b6f2b241f2bbf45d156bd8c621e40bb2caed4d6967281a3',
    shared='fe35674de649414c520cc73812d485c93cde6aadbc05db0c17ab94ca883e8c1b',
    pyannote='f284cc72de08b0113daa8a5482e2a2dbea2bd6a89ea00e128a569036394eebb8')
SCOPE = ROOT/'tests/parakeet/rational-sigmoid-results/graph-scope-20260927.json'


def previous_closed():
    verify_scope(); reports = {}
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
