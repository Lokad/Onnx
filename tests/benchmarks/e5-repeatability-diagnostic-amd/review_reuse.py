"""Verify the observer/source reuse before freezing any new VM campaign."""
import ast
import hashlib
import json
from pathlib import Path
from source_scope import instrument, verify

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'tests/benchmarks/e5-runtime-diagnostic-amd'
GRAPH = ROOT / 'artifacts/parakeet-slice-dense-conversion-graphs-amd-20260925'
TRACE = ROOT / 'artifacts/e5-runtime-diagnostic-amd-20260924'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def review():
    assert pin(GRAPH / 'closed.json')['sha256'] == 'e04e3a6a7434afd4af6bda1901c934004fc263db82a52f453f4321ca3f7a9fbd'
    graph = read(GRAPH / 'closed.json')
    assert graph['passed'] and not graph['admitted'] and not graph['all_controls_passed']
    assert pin(GRAPH / 'payload.json') == graph['files']['payload.json']
    payload = read(GRAPH / 'payload.json')
    original_consumer = payload['consumer']
    assert original_consumer['sha256'] == 'd827e3b9f1e5158e10bd24a5ca009fa7950fd08d08bb5260b84e36b5a853ac02'
    assert original_consumer == read(TRACE / 'payload.json')['previous_consumer']
    for case in ['e5-8tok', 'e5-512tok']:
        for role in ['current', 'candidate']:
            name = f'collected/timing-{case}-{role}-a/output/result.json'
            assert pin(GRAPH / name) == graph['files'][name]
            result = read(GRAPH / name)
            assert result['consumer'] == original_consumer['sha256']
            assert result['core'] == payload['products'][role]['Lokad.Onnx.dll']['sha256']
            assert result['calls'] == 780 and sum(row['warmup'] for row in result['clocks']) == 600
    previous = read(TRACE / 'prepared.json')
    source_path = ROOT / 'tests/benchmarks/warmed-release-amd-v2/Program.cs'
    assert pin(source_path) == previous['files'][source_path.relative_to(ROOT).as_posix()]
    source = source_path.read_text(encoding='utf8')
    actual = instrument(source)
    assert actual == (TRACE / 'bundle/source/consumer/Program.cs').read_text(encoding='utf8')
    source_review = verify(source, actual)
    for name in ['source_scope.py', 'Producer.csproj']:
        assert pin(HERE / name) == pin(OLD / name) == previous['files'][(OLD / name).relative_to(ROOT).as_posix()]
    name = 'il_normalization.py'
    assert pin(OLD / name) == previous['files'][(OLD / name).relative_to(ROOT).as_posix()]
    assert (HERE / name).read_text().rstrip() == (OLD / name).read_text().rstrip()
    old_probe = (OLD / 'ClockProbe.cs').read_text(encoding='utf8')
    assert pin(OLD / 'ClockProbe.cs') == previous['files'][(OLD / 'ClockProbe.cs').relative_to(ROOT).as_posix()]
    before = 'if (key != "e5-30tok" || mode != "timing")'
    after = 'if ((key != "e5-8tok" && key != "e5-512tok") || mode != "timing")'
    probe = (HERE / 'ClockProbe.cs').read_text(encoding='utf8')
    assert probe.count(after) == 1 and probe.replace(after, before) == old_probe
    old_export = ROOT / 'tests/parakeet/dispatch-events-amd/ExportAll.cs.txt'
    assert pin(old_export) == previous['files'][old_export.relative_to(ROOT).as_posix()]
    original = old_export.read_text(encoding='utf8')
    before = 'using var stream = new StreamWriter(new FileStream(Path.Combine(output,"events.jsonl"),FileMode.CreateNew));'
    after = 'using var stream = new StreamWriter(new System.IO.Compression.GZipStream(new FileStream(Path.Combine(output,"events.jsonl.gz"),FileMode.CreateNew), System.IO.Compression.CompressionLevel.Fastest));'
    exporter = (HERE / 'ExportAll.cs.txt').read_text(encoding='utf8')
    assert exporter.count(after) == 1 and exporter.replace(after, before) == original
    transfers = [read(ROOT / f'artifacts/e5-runtime-event-transfers-20260924/{role}.json') for role in 'abcd']
    assert all(row['passed'] for row in transfers)
    def function(path, name):
        tree = ast.parse(path.read_text(encoding='utf8'))
        node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        return ast.dump(node, include_attributes=False)
    assert function(HERE / 'checks.py', 'compiled_scope') == function(OLD / 'checks.py', 'compiled_scope')
    assert function(HERE / 'audit.py', 'reconcile') == function(OLD / 'audit.py', 'reconcile')
    associations = ROOT / 'tests/benchmarks/e5-runtime-diagnostic-results/associations.py'
    published = read(associations.parent / 'observations-20260924.json')
    assert pin(associations) == published['inputs'][str(associations.relative_to(ROOT))]
    return dict(passed=True, source_review=source_review, original_consumer=original_consumer,
                products=payload['products'], observer_key_change_only=True, exporter_sink_change_only=True,
                compiled_scope_unchanged=True, event_reconciliation_unchanged=True,
                prior_export_bytes=[row['raw']['bytes'] for row in transfers],
                prior_compressed_bytes=[row['archive']['bytes'] for row in transfers],
                exporter_roundtrip_required=True, diagnostic_built=False, diagnostic_run=False,
                release_admitted=False)


if __name__ == '__main__':
    print(json.dumps(review(), indent=2))
