"""Independently audit the offline payload without model execution or VM access."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-amd-candidates-v2-20260921'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    read = lambda p: json.loads(p.read_text(encoding='utf8'))
    prepared = read(BASE/'prepared.json')
    payload = BASE/'payload'
    assert prepared['passed'] is True
    assert pin(payload/'payload.json') == prepared['payload']
    assert pin(BASE/'payload.tar.gz') == prepared['archive']
    spec = read(payload/'payload.json')
    assert len(spec['files']) == prepared['files'] == 1334
    assert sum(v['bytes'] for v in spec['files'].values()) == prepared['bytes']
    for name, wanted in spec['files'].items():
        path = (payload/name).resolve()
        assert path.is_relative_to(payload.resolve()) and pin(path) == wanted, name
    expected = dict(spec['files'], **{'payload.json': prepared['payload']})
    seen = set()
    with tarfile.open(BASE/'payload.tar.gz') as archive:
        for entry in archive:
            assert entry.isfile() and entry.name not in seen and entry.name in expected, entry.name
            seen.add(entry.name)
            with archive.extractfile(entry) as stream:
                actual = dict(bytes=entry.size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
            assert actual == expected[entry.name], entry.name
    assert seen == set(expected)
    for name, wanted in prepared['predecessors'].items():
        assert pin(ROOT/name) == wanted, name
    failure = read(BASE/'predecessor-failure.json')
    assert pin(ROOT/failure['path']) == {k: failure[k] for k in ('bytes', 'sha256')}
    assert read(ROOT/failure['path'])['inference_executed'] is False
    rows = ROOT/'artifacts/pyannote-conv-row-sharing-v2-20260921/candidate-source'
    for name, wanted in spec['source_files'].items():
        assert pin(rows/name) == pin(payload/'source'/name) == wanted, name
    adaptation = spec['graph_consumer_adaptation']
    original = ROOT/'artifacts/pyannote-spatial-panels-20260921/consumer/Program.cs'
    assert pin(original) == adaptation['original']
    assert (payload/'graph-consumer/Program.cs').read_text() == original.read_text().replace(
        adaptation['old_guard'], adaptation['new_guard'])
    assert pin(BASE/'logs/local-il-bridge.json') == prepared['il_bridge']
    bridge = read(BASE/'logs/local-il-bridge.json')
    assert bridge['passed'] is True and len(bridge['observations']) == 2
    methods = {}
    for row in bridge['observations']:
        name = row['assembly']
        assert row['equal'] is True and row['methods'] == len(row['normalized_methods']) > 0
        assert row['before_sha256'] == pin(payload/'runtimes/rows'/name)['sha256']
        assert row['after_sha256'] == pin(payload/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name)['sha256']
        methods[name] = row['methods']
    builds = read(BASE/'local-builds.json')
    assert [r['name'] for r in builds] == [n+s for n in ('backend', 'tensors', 'graph-consumer', 'il-bridge')
                                         for s in ('-restore', '-build')]+['il-bridge']
    assert all(r['code'] == 0 for r in builds)
    for row in builds[:-1]:
        assert '--tl:off' in row['command']
        if row['name'].endswith('-restore'):
            assert '--source' in row['command'] and '--packages' in row['command']
    receipt = dict(passed=True, prepared=pin(BASE/'prepared.json'), payload=prepared['payload'],
                   archive=prepared['archive'], verified_files=len(seen), source_files=len(spec['source_files']),
                   compiled_methods=methods, offline_commands=len(builds),
                   amd_execution=False, numerical_qualification=False, performance_qualification=False)
    with (BASE/'preparation-verified.json').open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
