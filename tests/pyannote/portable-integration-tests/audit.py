"""Independently close the test-only successor after its controller exits."""
from common import *


def main():
    assert not (BASE / 'closed.json').exists()
    prepared, initial, state = (read(BASE / name) for name in ['prepared.json', 'source-prepared.json', 'processes.json'])
    assert prepared['passed'] and initial['passed'] and state['complete'] and state['code'] == 0
    verify(prepared['files'])
    verify(initial['files'])
    verify(prepared['products'])
    assert prepared['products'] == initial['products']
    previous = read(COMPLETE / 'closed.json')
    assert pin(COMPLETE / 'closed.json')['sha256'] == 'c07358dee34e3983ab7c212488772deaf0c973cc45274c5b505ac766e27753b4'
    assert previous['passed']
    verify(previous['files'])
    source = BASE / 'source'
    reference = PRIOR / 'evidence-inputs/src/Lokad.Onnx.Data/WeSpeakerAudio.cs'
    dense = reference.read_text(encoding='utf-8-sig')
    expected = dense.replace('namespace Lokad.Onnx;', 'namespace Lokad.Onnx.Backend.Tests;').replace(
        'public static class WeSpeakerAudio', 'internal static class DenseWeSpeakerReference')
    assert expected == (source / 'tests/Lokad.Onnx.Backend.Tests/DenseWeSpeakerReference.cs').read_text(encoding='utf8')
    assert 'MelSupport' not in dense and 'k < FourierSize / 2; k++' in dense
    before = (PRIOR / 'source/tests/Lokad.Onnx.Backend.Tests/SparseMelTests.cs').read_text(encoding='utf8')
    after = (source / 'tests/Lokad.Onnx.Backend.Tests/SparseMelTests.cs').read_text(encoding='utf8')
    # Keep every numerical and ownership assertion after Cases unchanged.
    assert before[before.index('    public static IEnumerable<object[]> Cases()'):] == after[after.index('    public static IEnumerable<object[]> Cases()'):]
    assert 'static readonly Frontend Dense = DenseWeSpeakerReference.LogMelFilterbank;' in after
    assert not any(token in after for token in ['AssemblyLoadContext', 'LoadDense', 'dense-reference'])
    assert not list(source.rglob('dense-reference'))
    changed = []
    for path in source.rglob('*'):
        if not path.is_file() or {'bin', 'obj'}.intersection(path.relative_to(source).parts):
            continue
        original = PRIOR / 'source' / path.relative_to(source)
        if not original.exists() or pin(path) != pin(original):
            changed.append(path.relative_to(source).as_posix())
    assert sorted(changed) == ['tests/Lokad.Onnx.Backend.Tests/DenseWeSpeakerReference.cs', 'tests/Lokad.Onnx.Backend.Tests/SparseMelTests.cs']
    backend = source / 'tests/Lokad.Onnx.Backend.Tests'
    project = (backend / 'Lokad.Onnx.Backend.Tests.csproj').read_text()
    assert '<ProjectReference' in project and '<HintPath>' not in project
    for name, field in [('Lokad.Onnx.dll', 'core'), ('Lokad.Onnx.Data.dll', 'data')]:
        assert pin(backend / 'bin/Release/net10.0' / name) == prepared[field] == initial[field] == pin(PRIOR / 'runtime' / name)
    suites = [read_suite(*args) for args in [('focused', 89, 0), ('hardware-disabled', 89, 0), ('backend-full', 3290, 93), ('tensors-full', 342, 0)]]
    assert suites == prepared['suites'] == read(BASE / 'suites.json')
    assert [r['name'] for r in state['runs']] == ['backend-build', 'focused', 'hardware-disabled', 'backend-full', 'tensors-full']
    assert '-p:BuildProjectReferences=false' in state['runs'][0]['command']
    identities, resources = [state['supervisor']], []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
        assert run['preflight']['available'] >= (8 if run['name'] == 'backend-build' else 10) * 1024**3
        samples = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(row['rss'] for row in samples) == run['peak_rss']
        for row in samples:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3
            assert row['disk'] >= 20 * 1024**3 and row['output_bytes'] <= 1024**3
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        resources.append(dict(name=run['name'], seconds=run['seconds'], samples=len(samples), peak_rss=run['peak_rss']))
    for identity in identities + previous['identities']:
        terminal(identity)
    analysis = dict(passed=True, changed_test_files=changed, normal_project_references=True,
        dense_reference_source=pin(reference), dense_reference_namespace_and_type_only=True,
        prior_numerical_assertions_preserved=True, historical_dll_absent=True, products_unchanged=True,
        core=prepared['core'], data=prepared['data'], suites=suites, identities=identities, resources=resources,
        resource_samples=sum(row['samples'] for row in resources), peak_rss=max(row['peak_rss'] for row in resources), scope=prepared['scope'])
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and 'obj' not in path.relative_to(BASE).parts:
            files[rel(path)] = pin(path)
    save(BASE / 'closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'analysis.json'), identities=identities))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), resource_samples=analysis['resource_samples'], changed=changed)))


if __name__ == '__main__':
    main()
