"""Bind the two failed inputs and unchanged products to the existing observer."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save, LIMITS
from review_reuse import review
from source_scope import instrument, verify

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/e5-repeatability-diagnostic-amd-20260925'
GRAPH = ROOT / 'artifacts/parakeet-slice-dense-conversion-graphs-amd-20260925'
TRACE = ROOT / 'artifacts/e5-runtime-diagnostic-amd-20260924'
WARM = ROOT / 'artifacts/warmed-release-amd-v2-20260923'
APP = ROOT / 'artifacts/parakeet-packed-final-row-app-amd-20260925'
REMOTE_GRAPH = '/dev/shm/lokad-parakeet-slice-dense-conversion-graphs-20260925'
REMOTE_TRACE = '/dev/shm/lokad-e5-runtime-diagnostic-20260924'
REMOTE_WARM = '/dev/shm/lokad-warmed-release-v2-20260923'
REMOTE_APP = '/dev/shm/lokad-parakeet-packed-final-row-app-20260925'


def previous_closed():
    reused = review()
    for folder in [GRAPH, TRACE, WARM, APP]:
        proof = read(folder / 'closed.json')
        assert proof['passed']
        for name, wanted in proof['files'].items():
            assert pin(folder / name) == wanted, name
    # This explanation remains necessary for M73 regardless of M78 admission.
    failed = [dict(key=r['key'], **c) for r in read(GRAPH / 'analysis.json')['performance']
              for c in r['controls'] if not c['passed']]
    assert [(r['key'], r['role']) for r in failed] == [('e5-8tok', 'candidate'), ('e5-512tok', 'candidate')]
    assert read(APP / 'collected/collection.json')['terminal']
    old = read(TRACE / 'prepared.json')['files']
    for name in ['tests/benchmarks/release-amd-v2/NpySupport.cs', 'global.json',
                 'tests/parakeet/dispatch-events-amd/remote.py']:
        assert pin(ROOT / name) == old[name], name
    assert pin(TOOLS / 'Exporter.csproj') == old['tests/parakeet/dispatch-events-amd/Exporter.csproj']
    return reused, failed


def prepare():
    assert not BASE.exists()
    reused, failed = previous_closed()
    BASE.mkdir()
    bundle = BASE / 'bundle'
    bundle.mkdir()
    originals = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    for name in ['ClockProbe.cs', 'Producer.csproj']:
        copy(TOOLS / name, bundle / 'source/consumer' / name)
    source = ROOT / 'tests/benchmarks/warmed-release-amd-v2/Program.cs'
    copy(source, bundle / 'evidence/OriginalProgram.cs.txt')
    original = source.read_text(encoding='utf8')
    actual = instrument(original)
    (bundle / 'source/consumer/Program.cs').write_text(actual, encoding='utf8')
    save(bundle / 'evidence/source-review.json', verify(original, actual))
    save(bundle / 'evidence/reuse-review.json', reused)
    copy(ROOT / 'tests/benchmarks/release-amd-v2/NpySupport.cs', bundle / 'source/consumer/NpySupport.cs')
    copy(TOOLS / 'ExportAll.cs.txt', bundle / 'source/exporter/Export.cs')
    copy(TOOLS / 'Exporter.csproj', bundle / 'source/exporter/Exporter.csproj')
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'il_normalization.py']:
        copy(TOOLS / name, bundle / 'tools' / name)
    copy(ROOT / 'tests/parakeet/dispatch-events-amd/remote.py', bundle / 'tools/remote_base.py')
    copy(TOOLS / 'README.md', bundle / 'README.md')
    shutil.copy2(ROOT / '.agent/m76-e5-repeatability-diagnostic-20260925.md', bundle / 'prospective-plan.md')
    receipts = {}
    for label, folder, remote in [('graph', GRAPH, REMOTE_GRAPH), ('trace', TRACE, REMOTE_TRACE),
                                  ('bridge', WARM, REMOTE_WARM), ('application', APP, REMOTE_APP)]:
        copy(folder / 'closed.json', bundle / 'evidence' / (label + '-closed.json'))
        copy(folder / 'collected/collection.json', bundle / 'evidence' / (label + '-collection.json'))
        receipts[label] = remote + '/collection.json'
    copy(GRAPH / 'analysis.json', bundle / 'evidence/graph-analysis.json')
    payload = read(GRAPH / 'payload.json')
    trace_payload = read(TRACE / 'payload.json')
    graph_files = read(GRAPH / 'collected/collection.json')['files']
    trace_files = read(TRACE / 'collected/collection.json')['files']
    links = {}
    external = dict(trace_payload['external'])
    for role in ['current', 'candidate']:
        manifest = read(GRAPH / f'collected/cases-{role}.json')
        cases = [c for c in manifest['cases'] if c['key'] in ['e5-8tok', 'e5-512tok']]
        assert [c['key'] for c in cases] == ['e5-8tok', 'e5-512tok']
        manifest['cases'] = cases
        save(bundle / f'cases-{role}.json', manifest)
        for name, wanted in graph_files.items():
            if name.startswith(f'runtimes/{role}/') and not Path(name).name.startswith('ReleaseBenchmark.'):
                links[name] = dict(source=REMOTE_GRAPH + '/' + name, identity=wanted)
        for case in cases:
            copy(GRAPH / f'collected/timing-{case["key"]}-{role}-a/output/result.json',
                 bundle / f'evidence/original-{case["key"]}-{role}.json')
            external[case['model']] = payload['external'][case['model']]
            for row in [*case['inputs'], *case['outputs']]:
                if 'file' in row:
                    name = row['file']
                    links[name] = dict(source=REMOTE_GRAPH + '/' + name, identity=payload['files'][name])
    for name, wanted in trace_payload['files'].items():
        if name.startswith('tracer/'):
            links[name] = dict(source=REMOTE_TRACE + '/' + name, identity=wanted)
    for name, wanted in read(WARM / 'collected/collection.json')['files'].items():
        if name.startswith('bridge/'):
            links[name] = dict(source=REMOTE_WARM + '/' + name, identity=wanted)
    for name, wanted in graph_files.items():
        if name.startswith('runtimes/current/'):
            links[name.replace('runtimes/current/', 'previous/')] = dict(source=REMOTE_GRAPH + '/' + name, identity=wanted)
    transferred = ROOT / 'artifacts/e5-runtime-event-transfers-20260924/a.json'
    copy(transferred, bundle / 'evidence/roundtrip-original.json')
    roundtrip = read(transferred)
    assert roundtrip['passed']
    # All four VM trace copies were retired after their original audit. Transfer
    # the complete retained local trace; never infer existence from an old receipt.
    copy(TRACE / 'collected/a-capture/capture.nettrace', bundle / 'roundtrip/capture.nettrace')
    assert pin(bundle / 'roundtrip/capture.nettrace') == trace_files['a-capture/capture.nettrace']
    # Retain every event. Estimate from complete prior traces before freezing;
    # the existing hard stage limit still rejects unexpected growth.
    trace_max = max(v['bytes'] for n, v in trace_files.items() if n.endswith('capture.nettrace'))
    compressed_max = max(reused['prior_compressed_bytes'])
    input_bytes = sum(r['identity']['bytes'] for r in links.values()) + sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    estimate = input_bytes + 8 * (trace_max + compressed_max) + compressed_max + 96 * 1024**2
    assert estimate < LIMITS['artifacts'], (estimate, LIMITS['artifacts'])
    save(bundle / 'stage.json', dict(passed=True, links=links, receipts=receipts, products=payload['products'],
        external=external, previous_consumer=graph_files['runtimes/current/ReleaseBenchmark.dll'],
        feed=trace_payload['feed'], interpreter=trace_payload['interpreter'], failed_release_controls=failed,
        roundtrip=dict(raw=roundtrip['raw'], events=roundtrip['events']),
        storage_estimate=dict(bytes=estimate, input_bytes=input_bytes, prior_max_trace=trace_max,
                              prior_max_compressed=compressed_max, allowance=96 * 1024**2),
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf8'), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals, stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(passed=True, archive=pin(BASE / 'payload.tar.gz'), links=len(links), estimated_bytes=estimate)))


if __name__ == '__main__':
    prepare()
