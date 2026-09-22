"""Preserve the root policy failure; finish corrected tensor and package checks."""
import importlib.util
from pathlib import Path
import shutil
import traceback
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-portable-root-completion-20260922'
PRIOR = ROOT / 'artifacts/pyannote-portable-root-integration-20260922'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('root_completion_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()


def suite(path, passed, skipped):
    root = ET.parse(path).getroot()
    rows = root.findall('.//{*}UnitTestResult')
    counters = root.find('.//{*}Counters').attrib
    assert len(rows) == int(counters['total']) == passed + skipped
    assert sum(r.attrib['outcome'] == 'Passed' for r in rows) == int(counters['passed']) == passed
    assert sum(r.attrib['outcome'] == 'NotExecuted' for r in rows) == skipped
    assert all(r.attrib['outcome'] in ('Passed', 'NotExecuted') for r in rows)
    return dict(passed=passed, skipped=skipped, trx=pin(path))


def predecessor():
    assert pin(PRIOR / 'failure-closed.json')['sha256'] == 'cbe3db5a526f4614797d44eb5f3b35c52ca9aa2dc30c4df34477faef33b47431'
    proof = read(PRIOR / 'failure-closed.json')
    assert proof['audited'] and proof['product_instructions_equal'] and proof['tensor_failed'] == 1
    for name, wanted in proof['files'].items():
        assert pin(PRIOR / name) == wanted, name
    for identity in proof['identities']:
        terminal(identity)
    verify(read(PRIOR / 'inputs.json')['files'])
    for row in read(PRIOR / 'instructions.json')['observations']:
        assert row['public_surface_equal'] and not row['differences'] and not row['added'] and not row['removed']
        assert pin(PRIOR / 'runtime' / row['assembly'])['sha256'] == row['after_sha256']
    return proof


def main():
    assert not BASE.exists()
    predecessor()
    BASE.mkdir(); (BASE / 'logs').mkdir()
    names = read(PRIOR / 'admission.json')['names'] + ['tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs']
    assert len(set(names)) == 24
    files = dict(read(PRIOR / 'inputs.json')['files'])
    for p in [*[ROOT / n for n in names], *TOOLS.glob('*.py'), MONITOR, PRIOR / 'failure-closed.json']:
        files[rel(p)] = pin(p)
    save(BASE / 'inputs.json', dict(passed=True, files=files, source_names=names, predecessor=pin(PRIOR / 'failure-closed.json')))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']

    def run(name, command, inference, output):
        monitor.worker(state, BASE / 'processes.json', name, command, ROOT, [0], 10 if inference else 8, 8, 900, True, output)
        print(name, 'passed', flush=True)

    try:
        project = ROOT / 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
        run('tensor-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers',
            '-p:BuildProjectReferences=false'], False, project.parent / 'bin')
        assert pin(project.parent / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(PRIOR / 'runtime/Lokad.Onnx.dll')
        run('tensor-tests', ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
            '--logger', 'trx;LogFileName=tensors.trx', '--results-directory', BASE / 'test-results'], True, BASE / 'test-results')
        tensors = suite(BASE / 'test-results/tensors.trx', 343, 0)
        run('package', ['dotnet', 'pack', ROOT / 'src/Lokad.Onnx/Lokad.Onnx.csproj', '-c', 'Release', *flags,
            '--no-build', '--no-restore', '--output', BASE / 'nuget'], False, BASE / 'nuget')
        package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
        with zipfile.ZipFile(package) as archive:
            assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (PRIOR / 'runtime/Lokad.Onnx.dll').read_bytes()
        consumer = BASE / 'consumer'; consumer.mkdir()
        project = consumer / 'PackageProbe.csproj'
        project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework></PropertyGroup><ItemGroup><PackageReference Include="Lokad.Onnx" Version="0.2.0" /></ItemGroup></Project>\n', encoding='utf8')
        shutil.copy2(ROOT / 'tests/pyannote/portable-integration/PackageProbe.cs', consumer / 'Program.cs')
        run('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', BASE / 'nuget', '--source', FEED,
            '--packages', BASE / 'consumer-cache'], False, None)
        run('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, consumer)
        assert pin(consumer / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(PRIOR / 'runtime/Lokad.Onnx.dll')
        run('consumer', ['dotnet', consumer / 'bin/Release/net10.0/PackageProbe.dll',
            ROOT / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx', pin(PRIOR / 'runtime/Lokad.Onnx.dll')['sha256'], BASE / 'consumer.json'], True, consumer)
        assert read(BASE / 'consumer.json')['passed']
        verify(files); predecessor()
        backend = suite(PRIOR / 'test-results/backend.trx', 3290, 93)
        save(BASE / 'verified.json', dict(passed=True, files=files, backend=backend, tensors=tensors,
            core=pin(PRIOR / 'runtime/Lokad.Onnx.dll'), data=pin(PRIOR / 'runtime/Lokad.Onnx.Data.dll'),
            package=pin(package), instructions=pin(PRIOR / 'instructions.json'), no_new_performance_measurement=True,
            policy_correction='Only two exact immutable historical fixture paths/content hashes are recognized; all other source is scanned. Mutation/other-path regression test added.'))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
