"""Independently verify every public Parakeet/native prerequisite and timed request."""
from candidate_protocol import pin, read, ROLES


def inspect(base, campaign, label, role, family, mode):
    from protocol import validate_records
    path = base / 'manifests' / ((ROLES[0] if role == 'ort' else role) + '-' + family + '.json')
    manifest = read(path); folder = campaign / (label + '-output'); result = read(folder / 'result.json')
    assert len(manifest['cases']) == (20 if family == 'parakeet' else 4)
    if family == 'parakeet': assert sum(c['samples'] for c in manifest['cases']) == 3412240
    validate_records(result, manifest, mode)
    assert result['manifest_sha256'] == pin(path)['sha256']
    assert result['engine'] == ('ort' if role == 'ort' else 'managed')
    for index, row in enumerate(result['records']): assert row == read(folder / f'{index:03}.json')
    if role == 'ort':
        spec = read(base / 'payload.json')
        assert result['python_binary'] == spec['interpreter'] and result['runner_sha256'] == pin(base / 'runtime/native.py')['sha256']
        assert result['versions'] == manifest['native_versions'] and result['native_binaries'] == manifest['native_binaries']
        assert result['native_settings'] == dict(provider='CPUExecutionProvider', intra_threads=1, inter_threads=1,
            sequential=True, graph_optimizations='all', spinning=False)
        assert result['numeric_libraries']
        for name, wanted in result['numeric_libraries'].items(): assert spec['external'][name] == wanted
    else:
        assert result['runtime'] == '.NET 10.0.8' and result['processor_count'] == 1
        assert result['runner_sha256'] == pin(base / 'runtimes' / role / 'AudioBenchmark.dll')['sha256']
        assert all(result[k] == manifest[k] for k in ('core_sha256', 'data_sha256'))
    return result


def conformance(base, campaign):
    rows = {}
    for role in (*ROLES, 'ort'):
        label = role + '-parakeet-public'
        result = inspect(base, campaign, label, role, 'parakeet', 'conformance')
        assert len(result['records']) == 20
        rows[role] = dict(calls=20, result=pin(campaign / (label + '-output/result.json')))
    native = inspect(base, campaign, 'native-conformance', 'ort', 'pyannote', 'conformance')
    assert len(native['records']) == 4
    return dict(passed=True, parakeet=rows, pyannote_native=pin(campaign / 'native-conformance-output/result.json'))
