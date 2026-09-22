"""Review the finite instruction changes, execute suites and consume the package."""
import shutil
import traceback
import zipfile
from common import *


def review():
    inventory = read(BASE / 'instructions.json')
    assert inventory['inventory_complete']
    observations = inventory['observations']
    assert [r['assembly'] for r in observations] == ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']
    for row in observations:
        assert row['public_surface_equal']
        assert row['before_sha256'] == pin(PRIOR / 'runtime' / row['assembly'])['sha256']
        assert row['after_sha256'] == pin(BASE / 'runtime' / row['assembly'])['sha256']
        if row['assembly'] == 'Lokad.Onnx.Data.dll':
            assert row['methods'] == row['unchanged_methods'] == 697
            assert not row['removed'] and not row['added'] and not row['differences']
            continue
        assert row['methods'] == 3108 and row['unchanged_methods'] == 3105
        prefix = 'Lokad.Onnx.Tensor`1[T]::'
        assert len(row['removed']) == 1 and row['removed'][0].startswith(prefix + 'RunTiledBatchFloat::')
        assert len(row['added']) == 4
        assert {k.split('::')[1] for k in row['added']} == {
            'RunTiledBatchFloat', 'CanUseConvPackedRows', 'ConvPackedScratchLength', 'TryConvPackedTile'}
        assert all(k.startswith(prefix) for k in row['added'])
        assert len(row['differences']) == 2
        assert sum(k.startswith(prefix + 'RunTiledConvFloat::') for k in row['differences']) == 1
        assert sum(k.startswith('Lokad.Onnx.Tensor`1+<>c__DisplayClass')
            and '::<RunTiledConvFloat>b__1::' in k for k in row['differences']) == 1
        method = next(v for k, v in row['candidate_methods'].items() if k.startswith(prefix + 'RunTiledBatchFloat::'))
        body = json.loads(method)
        calls = [r['operand'] for r in body['instructions'] if r['opcode'] in ('call', 'callvirt', 'newobj')]
        packed = next(i for i, s in enumerate(calls) if 'TryConvPackedTile(' in s)
        portable = next(i for i, s in enumerate(calls) if 'TryConvPortableRows(' in s)
        generic = next(i for i, s in enumerate(calls) if 'MatMul2D(' in s)
        assert packed < portable < generic
        assert sum('DenseTensor`1[System.Single]::Void .ctor(' in s for s in calls[packed:portable]) == 3
    return dict(passed=True, instructions=pin(BASE / 'instructions.json'), observations=[
        {k: r[k] for k in ['assembly', 'methods', 'unchanged_methods', 'removed', 'added', 'differences',
            'before_sha256', 'after_sha256', 'public_surface_equal', 'compiler_rename']} for r in observations],
        dispatch='AVX-512 first; unchanged portable wrapper construction and generic fallback')


def main():
    prepared = read(BASE / 'prepared.json')
    assert prepared['passed']
    verify(prepared['files'])
    state0 = read(BASE / 'processes.json')
    assert state0['complete'] and state0['code'] == 0
    terminal(state0['supervisor'])
    instruction_review = review()
    assert not (BASE / 'qualification-processes.json').exists()
    save(BASE / 'instruction-review.json', instruction_review)
    files = {rel(p): pin(p) for p in TOOLS.iterdir() if p.is_file()}
    files[rel(BASE / 'instruction-review.json')] = pin(BASE / 'instruction-review.json')
    probe = ROOT / 'tests/pyannote/portable-integration/PackageProbe.cs'
    files[rel(probe)] = pin(probe)
    save(BASE / 'qualification-inputs.json', dict(passed=True, files=files, preparation=pin(BASE / 'prepared.json')))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE / 'qualification-processes.json'
    save(path, state)
    source, runtime = BASE / 'source', BASE / 'runtime'
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    suites = []

    def run(name, args, inference, output):
        monitor.worker(state, path, name, args, source, [0], 10 if inference else 8, 8, 900, True, output)
        print(name, 'passed', flush=True)

    try:
        for name, pattern, disabled, passed, skipped in SUITES:
            original = monitor.clean_env
            if disabled:
                def environment():
                    env = original()
                    env['DOTNET_EnableHWIntrinsic'] = '0'
                    return env
                monitor.clean_env = environment
            try:
                project = source / ('tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
                    if name == 'tensors-full' else 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj')
                args = ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                    '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results']
                if pattern:
                    args += ['--filter', pattern]
                run(name, args, True, BASE / 'test-results')
            finally:
                monitor.clean_env = original
            suites.append(read_suite(name, passed, skipped))
            save(BASE / 'suites.json', suites)
        run('package', ['dotnet', 'pack', source / 'src/Lokad.Onnx/Lokad.Onnx.csproj', '-c', 'Release', *flags,
            '--no-build', '--no-restore', '--output', BASE / 'nuget'], False, BASE / 'nuget')
        package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
        with zipfile.ZipFile(package) as archive:
            assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (runtime / 'Lokad.Onnx.dll').read_bytes()
            import xml.etree.ElementTree as ET
            document = ET.fromstring(archive.read('Lokad.Onnx.nuspec'))
            dependencies = [n.attrib for n in document.iter() if n.tag.split('}')[-1] == 'dependency']
            assert len(dependencies) == 1 and dependencies[0]['id'] == 'Google.Protobuf' and dependencies[0]['version'] == '3.33.5'
            save(BASE / 'package.json', dict(passed=True, package=pin(package), entries=archive.namelist(),
                dependencies=dependencies, core=pin(runtime / 'Lokad.Onnx.dll')))
        consumer = BASE / 'package-consumer'
        consumer.mkdir()
        project = consumer / 'PackageProbe.csproj'
        project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework></PropertyGroup><ItemGroup><PackageReference Include="Lokad.Onnx" Version="0.2.0" /></ItemGroup></Project>\n', encoding='utf8')
        shutil.copy2(probe, consumer / 'Program.cs')
        run('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', BASE / 'nuget', '--source', FEED,
            '--packages', BASE / 'consumer-cache'], False, None)
        run('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, consumer)
        assert pin(consumer / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(runtime / 'Lokad.Onnx.dll')
        run('consumer', ['dotnet', consumer / 'bin/Release/net10.0/PackageProbe.dll',
            source / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx', pin(runtime / 'Lokad.Onnx.dll')['sha256'],
            BASE / 'consumer.json'], True, consumer)
        assert read(BASE / 'consumer.json')['passed']
        verify(files)
        verify(prepared['files'])
        for folder in [consumer, BASE / 'nuget']:
            for p in folder.rglob('*'):
                if p.is_file() and 'obj' not in p.relative_to(folder).parts:
                    files[rel(p)] = pin(p)
        save(BASE / 'qualified.json', dict(passed=True, files=files, suites=suites,
            core=prepared['core'], data=prepared['data'], package=pin(package),
            scope='Normal composition build, full local tests and separate NuGet consumer; AMD qualification pending.'))
        state['code'] = 0
        print(dict(qualified=pin(BASE / 'qualified.json'), package=pin(package)), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(path, state)


if __name__ == '__main__':
    main()
