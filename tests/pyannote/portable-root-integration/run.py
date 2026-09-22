"""Apply the reviewed product patch only after the fixed AMD comparison admits it."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/pyannote-portable-root-integration-20260922'
AMD = ROOT / 'artifacts/pyannote-portable-amd-execution-20260922'
CANDIDATE = ROOT / 'artifacts/pyannote-portable-integration-tests-20260922'
RUNTIME = ROOT / 'artifacts/pyannote-portable-applications-20260922/runtime'
REVIEW = ROOT / 'tests/pyannote/portable-integration-tests'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
spec = importlib.util.spec_from_file_location('root_integration_monitor', ROOT / 'tests/parakeet/packing-budgets/common.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def relative(path):
    return path.relative_to(ROOT).as_posix()


def suite(name, minimum, maximum_skipped):
    path = BASE / 'test-results' / (name + '.trx')
    root = ET.parse(path).getroot()
    rows = root.findall('.//{*}UnitTestResult')
    counters = root.find('.//{*}Counters').attrib
    assert all(r.attrib['outcome'] in ('Passed', 'NotExecuted') for r in rows)
    passed = sum(r.attrib['outcome'] == 'Passed' for r in rows)
    skipped = sum(r.attrib['outcome'] == 'NotExecuted' for r in rows)
    assert passed == int(counters['passed']) == int(counters['executed']) >= minimum
    assert skipped <= maximum_skipped and len(rows) == int(counters['total']) and int(counters['failed']) == 0
    return dict(name=name, passed=passed, skipped=skipped, counters=counters, trx=pin(path))


def main():
    assert not BASE.exists()
    closed = read(AMD / 'closed.json')
    assert closed['passed']
    for name, wanted in closed['files'].items():
        assert pin(AMD / name) == wanted, name
    analysis = read(AMD / 'analysis.json')
    assert analysis['passed'] and analysis['performance']['admitted'] and analysis['meetings']['passed']
    import sys
    sys.path.insert(0, str(ROOT / 'tests/pyannote/portable-amd-integration'))
    from admission import evaluate
    assert evaluate(analysis['table']) == analysis['performance']
    measured = read(ROOT / 'artifacts/pyannote-portable-amd-payload-20260922/payload/manifests/portable-pyannote.json')
    assert measured['core_sha256'] == pin(RUNTIME / 'Lokad.Onnx.dll')['sha256']
    assert measured['data_sha256'] == pin(RUNTIME / 'Lokad.Onnx.Data.dll')['sha256']
    assert analysis['timing_calls'] == 96 and analysis['measured'] == 72
    local = read(AMD / 'controller/state.json')
    assert local['complete'] and local['code'] == 0
    for identity in [local['supervisor']] + [r['child'] for r in local['stages']]:
        terminal(identity)
    assert pin(CANDIDATE / 'closed.json')['sha256'] == 'bf9822426cd75a86cc8c02c065005f0005e1a80b4ee63d5c53313fed145235f6'
    verify(read(CANDIDATE / 'closed.json')['files'])
    assert pin(RUNTIME.parent / 'closed.json')['sha256'] == '9d62d3d21a174e9be7105e4224b4b4b986b89dc6085ec60cc99b3aa2fdd7da77'
    verify(read(RUNTIME.parent / 'closed.json')['files'])
    assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True) == ''
    patch = REVIEW / 'integration.patch'
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, check=True)
    review = read(REVIEW / 'observations-20260922.json')
    assert pin(patch) == review['patch']
    names = [r['path'] for r in review['patch_files']]
    assert len(names) == len(set(names)) == 23
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    before = {}
    for name in names:
        path = ROOT / name
        before[name] = pin(path) if path.exists() else None
        if path.exists():
            target = BASE / 'before' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    evidence = dict(passed=True, amd_closure=pin(AMD / 'closed.json'), amd_analysis=pin(AMD / 'analysis.json'),
        patch=pin(patch), names=names, before=before, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save(BASE / 'admission.json', evidence)
    subprocess.run(['git', 'apply', str(patch)], cwd=ROOT, check=True)
    for name in names:
        assert (ROOT / name).read_text(encoding='utf-8-sig') == (CANDIDATE / 'source' / name).read_text(encoding='utf-8-sig'), name
    bridge = BASE / 'bridge'
    bridge.mkdir()
    shutil.copy2(ROOT / 'tests/pyannote/combined-avx512/Inventory.cs.txt', bridge / 'Program.cs')
    shutil.copy2(ROOT / 'artifacts/pyannote-portable-integration-20260922/bridge/Bridge.csproj', bridge / 'Bridge.csproj')
    files = {relative(ROOT / name): pin(ROOT / name) for name in names}
    for p in [Path(__file__).resolve(), REVIEW / 'integration.patch', REVIEW / 'observations-20260922.json',
        bridge / 'Program.cs', bridge / 'Bridge.csproj', ROOT / 'tests/parakeet/packing-budgets/common.py',
        ROOT / 'tests/pyannote/portable-integration/PackageProbe.cs']:
        files[relative(p)] = pin(p)
    save(BASE / 'inputs.json', dict(passed=True, files=files))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    suites = []

    def run(name, command, inference, output):
        monitor.worker(state, BASE / 'processes.json', name, command, ROOT, [0], 10 if inference else 8,
            8, 900, True, output)
        print(name, 'passed', flush=True)

    try:
        projects = {name: ROOT / path for name, path in [('cli', 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
            ('backend', 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
            ('tensors', 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]}
        projects['bridge'] = bridge / 'Bridge.csproj'
        for name, project in projects.items():
            run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], False, None)
            run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, None)
        runtime = BASE / 'runtime'
        shutil.copytree(projects['cli'].parent / 'bin/Release/net10.0', runtime)
        for name in ['cli', 'backend', 'tensors']:
            for assembly in ['Lokad.Onnx.dll'] + ([] if name == 'tensors' else ['Lokad.Onnx.Data.dll']):
                assert pin(projects[name].parent / 'bin/Release/net10.0' / assembly) == pin(runtime / assembly)
        run('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', RUNTIME,
            runtime, BASE / 'instructions.json'], False, bridge)
        inventory = read(BASE / 'instructions.json')
        assert inventory['inventory_complete']
        assert [(r['assembly'], r['methods']) for r in inventory['observations']] == [('Lokad.Onnx.dll', 3108), ('Lokad.Onnx.Data.dll', 697)]
        for row in inventory['observations']:
            assert row['public_surface_equal'] and not row['added'] and not row['removed'] and not row['differences']
        for name, project, minimum, skipped in [('backend', projects['backend'], 3290, 93), ('tensors', projects['tensors'], 342, 0)]:
            run(name + '-tests', ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results'], True, BASE / 'test-results')
            suites.append(suite(name, minimum, skipped))
        run('package', ['dotnet', 'pack', ROOT / 'src/Lokad.Onnx/Lokad.Onnx.csproj', '-c', 'Release', *flags,
            '--no-build', '--no-restore', '--output', BASE / 'nuget'], False, BASE / 'nuget')
        package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
        with zipfile.ZipFile(package) as archive:
            assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (runtime / 'Lokad.Onnx.dll').read_bytes()
        consumer = BASE / 'consumer'
        consumer.mkdir()
        project = consumer / 'PackageProbe.csproj'
        project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework></PropertyGroup><ItemGroup><PackageReference Include="Lokad.Onnx" Version="0.2.0" /></ItemGroup></Project>\n', encoding='utf8')
        shutil.copy2(ROOT / 'tests/pyannote/portable-integration/PackageProbe.cs', consumer / 'Program.cs')
        run('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', BASE / 'nuget', '--source', FEED,
            '--packages', BASE / 'consumer-cache'], False, None)
        run('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, consumer)
        assert pin(consumer / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(runtime / 'Lokad.Onnx.dll')
        run('consumer', ['dotnet', consumer / 'bin/Release/net10.0/PackageProbe.dll',
            ROOT / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx', pin(runtime / 'Lokad.Onnx.dll')['sha256'],
            BASE / 'consumer.json'], True, consumer)
        assert read(BASE / 'consumer.json')['passed']
        verify(files)
        save(BASE / 'verified.json', dict(passed=True, files=files, suites=suites, package=pin(package),
            core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
            instructions=pin(BASE / 'instructions.json'), no_new_performance_measurement=True))
        state['code'] = 0
        print(json.dumps(dict(verified=pin(BASE / 'verified.json'), suites=suites, package=pin(package))), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
