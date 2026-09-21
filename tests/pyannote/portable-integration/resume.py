"""Resume only unfinished tensors/package stages after the closed TRX-counter refusal."""
import shutil
import traceback
import zipfile
from phase_v2 import *

PRIOR = BASE
BASE = ROOT / 'artifacts/pyannote-portable-integration-completion-20260922'
monitor.BASE = BASE


def main():
    assert not BASE.exists()
    path = PRIOR / 'failure-closed.json'
    assert pin(path)['sha256'] == '22310d63dd7e720c95a61222095fcb71fbe86af826f0847907f288ef00b5afd3'
    proof = read(path)
    assert not proof['passed'] and proof['checker_refusal'] and proof['numerical_failures'] == 0
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    source, runtime = PRIOR / 'source', PRIOR / 'runtime'
    files = dict(proof['files'])
    files[rel(path)] = pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():
            files[rel(path)] = pin(path)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    save(BASE / 'continuation-prepared.json', dict(passed=True, files=files, predecessor_failure=pin(PRIOR / 'failure-closed.json'),
        scope='Reuse successful source builds, method/public declaration proof, 203 normal/109 hardware-off cases and 3290 backend passes/93 skips; execute only the unfinished stages.'))
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']

    def run(name, command, is_test=False, inference=False, output=None):
        verify(files)
        monitor.worker(state, BASE / 'processes.json', name, command, source, [0], 10 if is_test or inference else 8,
            8, 900, not inference, output)
        print(name, 'passed', flush=True)

    try:
        project = source / 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
        assert pin(project.parent / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(runtime / 'Lokad.Onnx.dll')
        run('tensors-full', ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
            '--logger', 'trx;LogFileName=tensors-full.trx', '--results-directory', BASE / 'test-results'],
            is_test=True, output=BASE / 'test-results')
        tensor_suite = read_suite(BASE, 'tensors-full', 342, 0)
        save(BASE / 'suites.json', read(PRIOR / 'failure-analysis.json')['suites'] + [tensor_suite])
        run('package', ['dotnet', 'pack', source / 'src/Lokad.Onnx/Lokad.Onnx.csproj', '-c', 'Release', *flags,
            '--no-build', '--no-restore', '--output', BASE / 'nuget'], output=BASE / 'nuget')
        package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
        with zipfile.ZipFile(package) as archive:
            assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (runtime / 'Lokad.Onnx.dll').read_bytes()
            assert 'Google.Protobuf' in archive.read('Lokad.Onnx.nuspec').decode('utf8')
            save(BASE / 'package.json', dict(passed=True, package=pin(package), entries=archive.namelist(), core=pin(runtime / 'Lokad.Onnx.dll')))
        consumer = BASE / 'package-consumer'
        consumer.mkdir()
        project = consumer / 'PackageProbe.csproj'
        project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework></PropertyGroup><ItemGroup><PackageReference Include="Lokad.Onnx" Version="0.2.0" /></ItemGroup></Project>\n', encoding='utf8')
        shutil.copy2(TOOLS / 'PackageProbeV2.cs', consumer / 'Program.cs')
        run('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', BASE / 'nuget', '--source', FEED,
            '--packages', BASE / 'consumer-cache'])
        run('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        assert pin(consumer / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(runtime / 'Lokad.Onnx.dll')
        run('consumer', ['dotnet', consumer / 'bin/Release/net10.0/PackageProbe.dll',
            source / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx', pin(runtime / 'Lokad.Onnx.dll')['sha256'],
            BASE / 'consumer.json'], inference=True, output=consumer)
        assert read(BASE / 'consumer.json')['passed']
        verify(files)
        state['code'] = 0
        for path in BASE.rglob('*'):
            if path.is_file() and not {'obj', 'consumer-cache'}.intersection(path.relative_to(BASE).parts) and path.name != 'processes.json':
                files[rel(path)] = pin(path)
        save(BASE / 'prepared.json', dict(passed=True, files=files, suites=read(BASE / 'suites.json'),
            core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'), package=pin(package),
            scope='Normal source build, exact method/public declarations, complete suites and local package consumption; original checker refusal retained. No new model timing or AMD/production promotion.'))
        print(dict(prepared=pin(BASE / 'prepared.json')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
