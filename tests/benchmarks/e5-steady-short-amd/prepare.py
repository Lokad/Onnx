"""Freeze the single measurement correction after exact-product runtime diagnosis."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import JOBS, LIMITS, pin, read, save
from scope import verify as scope

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/e5-steady-short-amd-20260925'
GRAPH = ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
DIAGNOSTIC = ROOT/'artifacts/e5-relocation-tier-diagnostic-amd-20260925'
BRIDGE = ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'
REMOTE_GRAPH = '/dev/shm/lokad-parakeet-owned-batch-isolation-graphs-20260925'
REMOTE_DIAGNOSTIC = '/dev/shm/lokad-e5-relocation-tier-diagnostic-20260925'
REMOTE_BRIDGE = '/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925'
CONSUMER = 'd827e3b9f1e5158e10bd24a5ca009fa7950fd08d08bb5260b84e36b5a853ac02'
CLOSURES = {
    GRAPH: 'def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638',
    DIAGNOSTIC: '5b87937aa914ab9007928bfb3ba0f90c99eae3ff8f3d23611467cbb98ab8ed4b',
    BRIDGE: 'e61941d9308e33b688f222665a0f79dd985f87b85d56123554a47202d8c531bc',
}


def previous_closed():
    scope()
    for folder, digest in CLOSURES.items():
        assert pin(folder/'closed.json')['sha256'] == digest
        closure = read(folder/'closed.json')
        assert closure['passed']
        for name, wanted in closure['files'].items():
            assert pin(folder/name) == wanted, name
    assert not read(GRAPH/'closed.json')['admitted']
    diagnostic = read(DIAGNOSTIC/'closed.json')
    assert diagnostic['diagnostic_only'] and not diagnostic['release_admitted']
    performance = read(GRAPH/'analysis.json')['performance']
    assert len(performance) == 8
    assert [r['key'] for r in performance if not r['qualified']] == ['e5-8tok']
    assert all(r['regression_passed'] for r in performance)
    graph, observed = read(GRAPH/'payload.json'), read(DIAGNOSTIC/'payload.json')
    assert graph['products'] == observed['products']
    assert graph['products']['current']['Lokad.Onnx.dll']['sha256'] == 'f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert graph['products']['candidate']['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert pin(GRAPH/'collected/runtimes/current/ReleaseBenchmark.dll')['sha256'] == CONSUMER
    source = ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs'
    assert pin(source) == pin(DIAGNOSTIC/'collected/evidence/OriginalProgram.cs.txt')
    assert source.read_text() == (TOOLS/'Program.cs').read_text().replace('3 : 6180', '3 : 780').replace('index < 6000', 'index < 600')


def prepare():
    previous_closed()
    assert not BASE.exists()
    BASE.mkdir()
    bundle = BASE/'bundle'
    bundle.mkdir()
    originals = scope()

    def copy(source, name):
        target = bundle/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'native.py']:
        copy(TOOLS/name, 'tools/'+name)
    copy(TOOLS/'Program.cs', 'source/consumer/Program.cs')
    for name in ['NpySupport.cs', 'ReleaseBenchmark.csproj']:
        copy(ROOT/'tests/benchmarks/release-amd-v2'/name, 'source/consumer/'+name)
    copy(ROOT/'global.json', 'source/global.json')
    copy(TOOLS/'README.md', 'README.md')
    shutil.copy2(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    copy(ROOT/'tests/benchmarks/e5-relocation-tier-results/interpretation-20260925.md', 'evidence/interpretation.md')
    receipts = {}
    for label, folder, remote in [('graph', GRAPH, REMOTE_GRAPH),
                                  ('diagnostic', DIAGNOSTIC, REMOTE_DIAGNOSTIC),
                                  ('bridge', BRIDGE, REMOTE_BRIDGE)]:
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder/name, 'evidence/'+label+'/'+name)
        copy(folder/'collected/collection.json', 'evidence/'+label+'/collection.json')
        receipts[label] = remote+'/collection.json'
    graph, diagnostic = read(GRAPH/'payload.json'), read(DIAGNOSTIC/'payload.json')
    cases = read(GRAPH/'collected/cases.json')
    cases['cases'] = [c for c in cases['cases'] if c['key'] == 'e5-8tok']
    assert len(cases['cases']) == 1
    save(bundle/'cases.json', cases)
    links = {}
    original_files = read(GRAPH/'collected/collection.json')['files']
    for name, wanted in original_files.items():
        if name.startswith('runtimes/current/'):
            links[name.replace('runtimes/current/', 'previous/')] = dict(source=REMOTE_GRAPH+'/'+name, identity=wanted)
        if name.startswith(('runtimes/current/', 'runtimes/candidate/')) and not Path(name).name.startswith('ReleaseBenchmark.'):
            links[name] = dict(source=REMOTE_GRAPH+'/'+name, identity=wanted)
    for name in ['Lokad.Onnx.dll', 'Google.Protobuf.dll']:
        key = 'runtimes/current/'+name
        links['product/'+name] = dict(source=REMOTE_GRAPH+'/'+key, identity=original_files[key])
    for row in [*cases['cases'][0]['inputs'], *cases['cases'][0]['outputs']]:
        if 'file' in row:
            name = row['file']
            links[name] = dict(source=REMOTE_GRAPH+'/'+name, identity=graph['files'][name])
    for name, wanted in read(BRIDGE/'collected/collection.json')['files'].items():
        if name.startswith('bridge/'):
            links[name] = dict(source=REMOTE_BRIDGE+'/'+name, identity=wanted)
    assert 'bridge/Bridge.dll' in links
    external = {name: wanted for name, wanted in graph['external'].items()
                if not name.startswith('/home/vermorel/Onnx/models/') or name == cases['cases'][0]['model']}
    for name, wanted in diagnostic['external'].items():
        assert external.setdefault(name, wanted) == wanted, name
    input_bytes = sum(v['identity']['bytes'] for v in links.values()) + sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    estimate = input_bytes + 128*1024**2
    assert estimate < LIMITS['artifacts']
    save(bundle/'stage.json', dict(passed=True, links=links, receipts=receipts,
        products=graph['products'], previous_consumer=pin(GRAPH/'collected/runtimes/current/ReleaseBenchmark.dll'),
        previous_owner=read(DIAGNOSTIC/'collected/collection.json')['identities'][0],
        feed=diagnostic['feed'], external=external, interpreter=graph['interpreter'],
        python_paths=graph['python_paths'], storage_estimate=estimate,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix == '.py':
                ast.parse(path.read_text(), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json'),
        links=len(links), jobs=len(JOBS), storage_estimate=estimate)))


if __name__ == '__main__':
    prepare()
