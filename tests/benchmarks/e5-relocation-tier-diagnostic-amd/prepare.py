"""Bind the existing observer to the closed relocation comparison, without building."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile
from protocol import LIMITS, pin, read, save
from consumer_scope import verify_scope

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'e5-direct-tier-diagnostic-v2-amd'
sys.path.insert(1, str(PARENT))
BASE = ROOT/'artifacts/e5-relocation-tier-diagnostic-amd-20260925'
TRACE = ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'
GRAPH = ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
BUILD = ROOT/'artifacts/parakeet-owned-batch-isolation-build-amd-20260925'
REMOTE_TRACE = '/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925'
REMOTE_GRAPH = '/dev/shm/lokad-parakeet-owned-batch-isolation-graphs-20260925'
OBSERVER = dict(bytes=36864, sha256='968d3beb8e19daa635af0c0e051f41f7f586e0a77a5a84120de5ba56c8c8b843')
CLOSURES = {
    TRACE:'e61941d9308e33b688f222665a0f79dd985f87b85d56123554a47202d8c531bc',
    GRAPH:'def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638',
    BUILD:'5dd53e90d4924a36bb6f43cc80fa939e9236492949542dcf659dc056d4ea3860',
}


def previous_closed():
    verify_scope()
    for folder, digest in CLOSURES.items():
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    graph = read(GRAPH/'analysis.json')
    assert not read(GRAPH/'closed.json')['admitted']
    assert all(row['regression_passed'] for row in graph['performance'])
    failed = [row for row in graph['performance'] if not row['qualified']]
    assert [row['key'] for row in failed] == ['e5-8tok']
    assert [c['role'] for c in failed[0]['controls'] if not c['passed']] == ['candidate']
    old, current = read(TRACE/'payload.json'), read(GRAPH/'payload.json')
    assert current['products']['current'] == old['products']['current']
    assert current['consumer'] == old['previous_consumer']
    assert current['products']['candidate']['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert current['products']['candidate']['Lokad.Onnx.dll'] == read(BUILD/'analysis.json')['product']['Lokad.Onnx.dll']
    assert read(BUILD/'build-review.json')['release_dispatcher_restored']
    built = read(TRACE/'collected/built.json'); assert built['passed'] and built['consumer'] == OBSERVER
    from checks import compiled_scope, observer_scope
    for name, checker in [('consumer-inventory', compiled_scope), ('observer-inventory', observer_scope)]:
        assert checker(read(TRACE/'collected'/name/'instructions.json'), old, built) == read(TRACE/'collected'/name/'review.json')
    assert old['reused_exporter']['roundtrip']['passed']
    assert built['exporter'] == old['reused_exporter']['binary']
    return failed


def prepare():
    failed = previous_closed(); assert not BASE.exists()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}

    def copy(source, name):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target); originals[source.relative_to(ROOT).as_posix()] = pin(source)

    for name in ['protocol.py', 'remote_prepare.py']:
        copy(TOOLS/name, 'tools/'+name)
    for name in ['remote.py', 'checks.py', 'source_scope.py', 'il_normalization.py']:
        copy(PARENT/name, 'tools/'+name)
    copy(ROOT/'tests/parakeet/dispatch-events-amd/remote.py', 'tools/remote_base.py')
    copy(TOOLS/'README.md', 'README.md')
    shutil.copy2(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    save(bundle/'evidence-scope.json', verify_scope())

    for label, folder in [('parent', TRACE), ('graph', GRAPH)]:
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder/name, 'evidence/'+label+'-'+name)
        copy(folder/'collected/collection.json', 'evidence/'+label+'-collection.json')
    copy(BUILD/'closed.json', 'evidence/build-closed.json')
    copy(BUILD/'build-review.json', 'evidence/build-review.json')
    trace_files = read(TRACE/'collected/collection.json')['files']
    graph_files = read(GRAPH/'collected/collection.json')['files']
    old, graph = read(TRACE/'payload.json'), read(GRAPH/'payload.json')
    for name, wanted in trace_files.items():
        if name.startswith(('consumer-inventory/', 'observer-inventory/')) or name in [
            'built.json', 'evidence/OriginalProgram.cs.txt', 'evidence/source-review.json',
            'source/consumer/Program.cs', 'source/consumer/ClockProbe.cs',
            'source/consumer/NpySupport.cs', 'source/consumer/Producer.csproj', 'source/global.json']:
            assert pin(TRACE/'collected'/name) == wanted
            copy(TRACE/'collected'/name, name)

    products, links = graph['products'], {}
    assert products['current']['Lokad.Onnx.dll']['sha256'] == 'f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    for role in ['current', 'candidate']:
        manifest = read(GRAPH/f'collected/cases-{role}.json')
        case, = [c for c in manifest['cases'] if c['key'] == 'e5-8tok']
        manifest['cases'] = [case]
        assert manifest['core'] == products[role]['Lokad.Onnx.dll']['sha256']
        assert case == read(TRACE/f'collected/cases-{role}.json')['cases'][0]
        save(bundle/f'cases-{role}.json', manifest)
        copy(GRAPH/f'collected/timing-e5-8tok-{role}-a/output/result.json',
             f'evidence/original-e5-8tok-{role}.json')
        for name, wanted in trace_files.items():
            if name.startswith('runtimes/current/'):
                target = name.replace('runtimes/current/', 'runtimes/'+role+'/')
                links[target] = dict(source=REMOTE_TRACE+'/'+name, identity=wanted)
        name = 'runtimes/'+role+'/Lokad.Onnx.dll'
        assert graph_files[name] == products[role]['Lokad.Onnx.dll']
        links[name] = dict(source=REMOTE_GRAPH+'/'+name, identity=graph_files[name])
        for row in [*case['inputs'], *case['outputs']]:
            if 'file' in row:
                name = row['file']; links[name] = dict(source=REMOTE_GRAPH+'/'+name, identity=graph['files'][name])
    for prefix, catalog in [('tracer/',old['files']), ('export-runtime/',trace_files)]:
        for name, wanted in catalog.items():
            if name.startswith(prefix): links[name] = dict(source=REMOTE_TRACE+'/'+name, identity=wanted)
    input_bytes = sum(value['identity']['bytes'] for value in links.values()) + sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    raw_max = max(trace_files[f'{r}-capture/capture.nettrace']['bytes'] for r in 'abcd')
    compressed_max = max(trace_files[f'{r}-export/events/events.jsonl.gz']['bytes'] for r in 'abcd')
    estimate = input_bytes + 4*(raw_max+compressed_max) + 64*1024**2
    assert estimate < LIMITS['artifacts'], (estimate, LIMITS['artifacts'])
    stage = dict(passed=True, links=links, products=products,
        receipts=dict(parent=REMOTE_TRACE+'/collection.json', graph=REMOTE_GRAPH+'/collection.json'),
        previous_owner=read(GRAPH/'collected/collection.json')['identities'][0],
        previous_consumer=old['previous_consumer'], previous_observer=old['previous_observer'],
        reused_consumer=OBSERVER, reused_exporter=old['reused_exporter'],
        parent_closure=pin(TRACE/'closed.json'), graph_closure=pin(GRAPH/'closed.json'),
        feed=old['feed'], interpreter=old['interpreter'], external=old['external'],
        failed_release_cases=failed, diagnostic_only=True, release_admitted=False,
        storage_estimate=dict(bytes=estimate, input_bytes=input_bytes, previous_max_trace=raw_max,
                              previous_max_compressed=compressed_max, allowance=64*1024**2),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    for folder in [TOOLS, PARENT]:
        for path in folder.iterdir():
            if path.is_file():
                if path.suffix == '.py': ast.parse(path.read_text(), str(path))
                originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'), links=len(links), estimated_bytes=estimate, observer=OBSERVER)))


if __name__ == '__main__':
    prepare()
