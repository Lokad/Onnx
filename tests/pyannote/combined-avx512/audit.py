"""Independently close the normal build, every test and package-consumer sample."""
import zipfile
from common import *
from qualify import review


def main():
    assert not (BASE / 'closed.json').exists()
    prepared, qualified = read(BASE / 'prepared.json'), read(BASE / 'qualified.json')
    initial, inputs = read(BASE / 'source-prepared.json'), read(BASE / 'qualification-inputs.json')
    for receipt in [prepared, qualified, initial, inputs]:
        assert receipt['passed']
        verify(receipt['files'])
    assert inputs['preparation'] == pin(BASE / 'prepared.json')
    assert review() == read(BASE / 'instruction-review.json')
    changed = []
    for p in sorted((BASE / 'source').rglob('*')):
        name = p.relative_to(BASE / 'source')
        if p.is_file() and not {'bin', 'obj', 'artifacts'}.intersection(name.parts):
            before = SOURCE / 'source' / name
            if not before.exists() or pin(p) != pin(before):
                changed.append(name.as_posix())
    assert changed == prepared['changed'] == initial['changed'] == [
        'src/Lokad.Onnx/TensorOps.ConvPool.cs', 'src/Lokad.Onnx/Zzz.ZConvPackedRows.cs',
        'tests/Lokad.Onnx.Backend.Tests/ConvPackedRowsTests.cs']
    assert pin(BASE / 'source/src/Lokad.Onnx/Zzz.ZConvPackedRows.cs') == pin(FEED.parent / 'source/src/Lokad.Onnx/TensorOps.ConvPackedRows.cs')
    suites = [read_suite(name, passed, skipped) for name, _, _, passed, skipped in SUITES]
    assert suites == qualified['suites'] == read(BASE / 'suites.json')
    expected = [name + '-' + stage for name in ['cli', 'backend', 'tensors', 'bridge'] for stage in ['restore', 'build']]
    builds = audit_resources(BASE / 'processes.json', expected + ['instructions'])
    tests = audit_resources(BASE / 'qualification-processes.json', [s[0] for s in SUITES]
        + ['package', 'consumer-restore', 'consumer-build', 'consumer'])
    consumer_run = read(BASE / 'qualification-processes.json')['runs'][-1]
    assert consumer_run['name'] == 'consumer' and consumer_run['preflight']['available'] >= 10 * 1024**3
    package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
    assert qualified['package'] == read(BASE / 'package.json')['package'] == pin(package)
    with zipfile.ZipFile(package) as archive:
        assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (BASE / 'runtime/Lokad.Onnx.dll').read_bytes()
    assert read(BASE / 'consumer.json')['passed']
    assert pin(BASE / 'package-consumer/bin/Release/net10.0/Lokad.Onnx.dll') == prepared['core']
    project = (BASE / 'package-consumer/PackageProbe.csproj').read_text()
    assert '<PackageReference Include="Lokad.Onnx" Version="0.2.0" />' in project and 'ProjectReference' not in project
    identities = builds['identities'] + tests['identities']
    resources = builds['resources'] + tests['resources']
    analysis = dict(passed=True, suites=suites, changed=changed, core=prepared['core'], data=prepared['data'],
        package=qualified['package'], instruction_review=pin(BASE / 'instruction-review.json'),
        identities=identities, resources=resources, resource_samples=sum(r['samples'] for r in resources),
        peak_rss=max(r['peak_rss'] for r in resources), scope=qualified['scope'])
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'], **qualified['files'])
    files[rel(Path(__file__).resolve())] = pin(Path(__file__).resolve())
    for p in BASE.rglob('*'):
        if p.is_file() and not {'obj', 'packages', 'consumer-cache'}.intersection(p.relative_to(BASE).parts):
            files[rel(p)] = pin(p)
    save(BASE / 'closed.json', dict(passed=True, files=files, identities=identities, analysis=pin(BASE / 'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), core=prepared['core'], data=prepared['data'],
        package=qualified['package'], resource_samples=analysis['resource_samples'], peak_rss=analysis['peak_rss'])))


if __name__ == '__main__':
    main()
