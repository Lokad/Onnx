"""Independently close source, IL/API, suites, package and process evidence."""
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from common import *


def main():
    assert not (BASE / 'closed.json').exists()
    prepared, state = read(BASE / 'prepared.json'), read(BASE / 'processes.json')
    assert prepared['passed'] and state['complete'] and state['code'] == 0
    verify(prepared['files'])
    inputs = read(BASE / 'source-inputs.json')
    verify(inputs['files'])
    assert len(inputs['product_overlays']) == 12 and len(inputs['added_tests']) == 10 and len(inputs['changed_text_files']) == 22
    source, runtime = BASE / 'source', BASE / 'runtime'
    for path in source.rglob('*.csproj'):
        if {'bin', 'obj'}.intersection(path.relative_to(source).parts):
            continue
        assert pin(path) == pin(ROOT / path.relative_to(source))
    instructions = read(BASE / 'instructions.json')
    assert instructions['passed'] and len(instructions['observations']) == 2
    method_rows = []
    for row in instructions['observations']:
        assert row['public_surface_equal'] and row['public_surface'] and not row['removed'] and row['equal_except_storage_admission']
        assert row['before_sha256'] == pin(APPLICATION / 'application-runtime' / row['assembly'])['sha256']
        assert row['after_sha256'] == pin(runtime / row['assembly'])['sha256']
        if row['assembly'] == 'Lokad.Onnx.dll':
            assert row['methods'] == 3107 and row['unchanged_methods'] == 3106 and len(row['differences']) == len(row['added']) == 1
            assert row['differences'][0].startswith('Lokad.Onnx.CPUExecutionProvider+LstmProjectionPanels::Create::')
            assert row['added'][0].startswith('Lokad.Onnx.CPUExecutionProvider+LstmProjectionPanels::StorageLength::')
        else:
            assert row['assembly'] == 'Lokad.Onnx.Data.dll' and row['methods'] == row['unchanged_methods'] == 697
            assert not row['differences'] and not row['added']
        method_rows.append({k:v for k,v in row.items() if k not in ['normalized_methods', 'candidate_methods', 'public_surface']})
        method_rows[-1]['public_declaration_records'] = len(row['public_surface'])
    expected = [('focused', 203, 0), ('hardware-disabled', 109, 0), ('backend-full', 3290, 93), ('tensors-full', 342, 0)]
    suites = [suite(*args) for args in expected]
    assert suites == prepared['suites'] == read(BASE / 'suites.json')
    jobs = ['cli-restore', 'cli-build', 'backend-restore', 'backend-build', 'tensors-restore', 'tensors-build',
        'bridge-restore', 'bridge-build', 'instructions', 'focused', 'hardware-disabled', 'backend-full', 'tensors-full',
        'package', 'consumer-restore', 'consumer-build', 'consumer']
    assert [r['name'] for r in state['runs']] == jobs
    identities, resources = [state['supervisor']], []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
        is_test = run['name'] in {r[0] for r in expected} or run['name'] == 'consumer'
        assert run['preflight']['available'] >= (10 if is_test else 8) * 1024**3
        samples = [json.loads(s) for s in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(r['rss'] for r in samples) == run['peak_rss']
        for row in samples:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3
            assert row['disk'] >= 20 * 1024**3 and row['output_bytes'] <= 1024**3
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
            assert run['name'] != 'consumer' or len(row['members']) <= 1
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        resources.append(dict(name=run['name'], seconds=run['seconds'], samples=len(samples), peak_rss=run['peak_rss']))
    for identity in identities:
        terminal(identity)
    package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
    assert pin(package) == prepared['package'] == read(BASE / 'package.json')['package']
    with zipfile.ZipFile(package) as archive:
        assert archive.namelist() == read(BASE / 'package.json')['entries']
        value = archive.read('lib/net10.0/Lokad.Onnx.dll')
        assert dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) == prepared['core'] == pin(runtime / 'Lokad.Onnx.dll')
        nuspec = ET.fromstring(archive.read('Lokad.Onnx.nuspec'))
        assert nuspec.find('.//{*}id').text == 'Lokad.Onnx' and nuspec.find('.//{*}version').text == '0.2.0'
        dependencies = [d.attrib for d in nuspec.findall('.//{*}dependency')]
        assert dependencies == [dict(id='Google.Protobuf', version='3.33.5', exclude='Build,Analyzers')], dependencies
        assert [g.attrib['targetFramework'] for g in nuspec.findall('.//{*}group')] == ['net10.0']
    consumer = read(BASE / 'consumer.json')
    assert consumer['passed'] and consumer['core'] == prepared['core']['sha256']
    assert consumer['runtime'] == '.NET 10.0.12' and consumer['processor_count'] == 1
    assert consumer['pid'] == state['runs'][-1]['worker']['pid']
    assert consumer['product'] == [5, 11, 14, 23] and consumer['convolution'] == [-3, 3, -7, 7] and consumer['activated'] == [0, 3, 0, 7]
    assert consumer['input_and_held_outputs_unchanged'] and consumer['model_imported']
    assert consumer['model'] == pin(source / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx')['sha256']
    assert consumer['executable'] == pin(BASE / 'package-consumer/bin/Release/net10.0/PackageProbe.dll')['sha256']
    assert Path(consumer['loaded_core_path']).resolve() == (BASE / 'package-consumer/bin/Release/net10.0/Lokad.Onnx.dll').resolve()
    project = ET.parse(BASE / 'package-consumer/PackageProbe.csproj')
    assert [p.attrib for p in project.findall('.//PackageReference')] == [dict(Include='Lokad.Onnx', Version='0.2.0')]
    assert not project.findall('.//ProjectReference') and not project.findall('.//Reference')
    analysis = dict(passed=True, root_commit=inputs['root_commit'], changed_source_files=inputs['changed_text_files'],
        normal_project_references=True, methods=method_rows, suites=suites, core=prepared['core'], data=prepared['data'],
        package=prepared['package'], package_dependencies=dependencies, consumer=consumer,
        resources=resources, resource_samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),
        identities=identities, scope=prepared['scope'])
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not {'obj', 'nuget-cache', 'consumer-cache'}.intersection(path.relative_to(BASE).parts):
            files[rel(path)] = pin(path)
    save(BASE / 'closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'analysis.json'), identities=identities))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), core=prepared['core'], data=prepared['data'], resources=analysis['resource_samples'])))


if __name__ == '__main__':
    main()
