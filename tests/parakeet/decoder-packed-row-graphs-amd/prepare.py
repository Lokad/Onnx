"""Bind the exact qualified eight-case graph protocol to the fixed prepared-row pair."""
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
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-graphs-amd-20260927'
GRAPH = ROOT/'artifacts/parakeet-rational-sigmoid-graphs-amd-20260927'
PRODUCT = ROOT/'artifacts/parakeet-decoder-packed-row-models-amd-20260927'
PRIOR = dict(graph=GRAPH, models=PRODUCT,
    app=ROOT/'artifacts/parakeet-decoder-packed-row-app-amd-20260927',
    shared=ROOT/'artifacts/parakeet-decoder-packed-row-shared-amd-20260927',
    pyannote=ROOT/'artifacts/parakeet-decoder-packed-row-pyannote-amd-20260927')
REMOTE = dict(graph='/dev/shm/lokad-parakeet-rational-sigmoid-graphs-20260927',
    models='/dev/shm/lokad-parakeet-decoder-packed-row-models-20260927',
    app='/dev/shm/lokad-parakeet-decoder-packed-row-app-20260927',
    shared='/dev/shm/lokad-parakeet-decoder-packed-row-shared-20260927',
    pyannote='/dev/shm/lokad-parakeet-decoder-packed-row-pyannote-20260927')
DIGESTS = dict(graph='167f01fdd3c8dce104377259ae4441a8e59c0a6bd6d666e457d38e3398153742',
    models='5d6832083d103bef9db7bb733b1e96decbae22625f3138680ab6f9a426c78e3b',
    app='26ac173f867a964b73761d3c716aa5db0c384909bb1849e2cc810ed90e0a1781',
    shared='bdd1c7664d902af7d135318f12856419a61d222e69bb073ae24e10b8aca8aca0',
    pyannote='74bd1f90716a11446846d12cbba344b501d2736777ac1d0b3eedd6eb10ca7a33')
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
