"""Build the reviewed candidate using normal project references, then consume its package."""
import difflib
import shutil
import subprocess
import traceback
import zipfile
from common import *


def main():
    assert not BASE.exists()
    files = prerequisites()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head.startswith('ad586f69')
    subprocess.run(['git', 'diff', '--exit-code', '--', 'src', 'tests/Lokad.Onnx.Backend.Tests',
        'tests/Lokad.Onnx.Tensors.Tests', 'tests/Shared'], cwd=ROOT, check=True)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source, runtime, bridge = BASE / 'source', BASE / 'runtime', BASE / 'bridge'
    source.mkdir()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf8').split('\0')
    copied = []
    for name in tracked:
        if not name:
            continue
        selected = '/' not in name or name.startswith(('src/', 'tests/Lokad.Onnx.Backend.Tests/',
            'tests/Lokad.Onnx.Tensors.Tests/', 'tests/Shared/')) or (name.startswith('tests/') and name.endswith('.csproj'))
        if selected:
            before, after = ROOT / name, source / name
            assert before.is_file()
            after.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(before, after)
            files[name] = pin(before)
            copied.append(name)
    inventory = read(INVENTORY / 'analysis.json')
    overlays = [row['path'] for row in inventory['rows'] if row['category'] in ['core-and-request-contexts', 'sparse-mel']]
    assert len(overlays) == 12 and all(name.endswith('.cs') for name in overlays)
    for name in overlays:
        before, after = ACCEPTED / 'source' / name, source / name
        files[rel(before)] = pin(before)
        after.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(before, after)
    additional = []
    for path in (ACCEPTED / 'source/tests/Lokad.Onnx.Backend.Tests').rglob('*.cs'):
        name = path.relative_to(ACCEPTED / 'source')
        if {'bin', 'obj'}.intersection(name.parts):
            continue
        if not (ROOT / name).exists():
            shutil.copy2(path, source / name)
            files[rel(path)] = pin(path)
            additional.append(name.as_posix())
        elif path.read_text(encoding='utf-8-sig') != (ROOT / name).read_text(encoding='utf-8-sig'):
            assert name.as_posix() == 'tests/Lokad.Onnx.Backend.Tests/CliExitCodeTests.cs'
    assert len(additional) == 8
    panel = 'src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs'
    before = (source / panel).read_text(encoding='utf-8-sig')
    after = (GUARD / panel).read_text(encoding='utf-8-sig')
    helper = '''
        internal static int StorageLength(int inputWeights, int recurrentWeights)
        {
            if (inputWeights < 0 || recurrentWeights < 0) return 0;
            long count = (long)inputWeights + recurrentWeights;
            return count <= Array.MaxLength ? (int)count : 0;
        }
'''
    expected = before.replace('            int count = checked(w.Length + r.Length);',
        '            int count = StorageLength(w.Length, r.Length);\n            if (count == 0) return null;')
    expected = expected.replace('        public void Dispose() => ArrayPool<float>.Shared.Return(storage);\n',
        '        public void Dispose() => ArrayPool<float>.Shared.Return(storage);\n' + helper)
    assert before != expected == after
    shutil.copy2(GUARD / panel, source / panel)
    files[rel(GUARD / panel)] = pin(GUARD / panel)
    for name in ['LstmPanelAdmissionTests.cs', 'LstmPanelOverflowRefusalTests.cs']:
        path = GUARD / 'tests/Lokad.Onnx.Backend.Tests' / name
        shutil.copy2(path, source / 'tests/Lokad.Onnx.Backend.Tests' / name)
        files[rel(path)] = pin(path)
    changes, patch = [], []
    for path in source.rglob('*'):
        if not path.is_file():
            continue
        name = path.relative_to(source).as_posix()
        original = ROOT / name
        if path.suffix in ['.cs', '.csproj']:
            old = original.read_text(encoding='utf-8-sig') if original.exists() else ''
            new = path.read_text(encoding='utf-8-sig')
            if old != new:
                changes.append(name)
                patch.append(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                    fromfile='a/' + name if original.exists() else '/dev/null', tofile='b/' + name)))
            if path.suffix == '.csproj':
                assert pin(path) == pin(original)
                if name.startswith(('src/', 'tests/Lokad.Onnx.Backend.Tests/', 'tests/Lokad.Onnx.Tensors.Tests/')):
                    assert '<HintPath>' not in new
    assert len(changes) == 22
    (BASE / 'candidate.patch').write_text(''.join(patch), encoding='utf8')
    save(BASE / 'source-inputs.json', dict(root_commit=head, copied=copied, product_overlays=overlays,
        added_tests=additional + ['tests/Lokad.Onnx.Backend.Tests/' + n for n in ['LstmPanelAdmissionTests.cs', 'LstmPanelOverflowRefusalTests.cs']],
        changed_text_files=changes, normal_project_references=True, files=files))
    shutil.copytree(ACCEPTED / 'bridge', bridge, ignore=shutil.ignore_patterns('bin', 'obj'))
    path = bridge / 'Program.cs'
    text = path.read_text(encoding='utf8')
    start, end = text.index('    const string audio ='), text.index('    if (!allowed)')
    text = text[:start] + '''    const string panel = "Lokad.Onnx.CPUExecutionProvider+LstmProjectionPanels::";
    bool allowed = name == "Lokad.Onnx.dll"
        ? removed.Length == 0 && added.Length == 1 && differences.Length == 1
          && added[0].StartsWith(panel + "StorageLength::")
          && differences[0].StartsWith(panel + "Create::")
        : removed.Length == 0 && added.Length == 0 && differences.Length == 0;
''' + text[end:]
    text = text.replace('equal_except_sparse_mel', 'equal_except_storage_admission')
    surface = '''string[] PublicSurface(Assembly assembly)
{
    var result = new List<string>();
    foreach (var type in assembly.GetExportedTypes())
    {
        result.Add($"TYPE {type.FullName} {type.Attributes} {type.BaseType}");
        result.AddRange(type.GetInterfaces().Select(t => $"INTERFACE {type.FullName} {t}"));
        foreach (var parameter in type.GetGenericArguments())
            result.Add($"GENERIC {type.FullName} {parameter} {parameter.GenericParameterAttributes} {string.Join(",", parameter.GetGenericParameterConstraints().Select(t => t.ToString()).Order())}");
        foreach (var member in type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly))
        {
            string key = $"MEMBER {type.FullName} {member.MemberType} {member}";
            result.Add(key);
            result.AddRange(member.GetCustomAttributesData().Select(a => key + " ATTRIBUTE " + a));
            if (member is MethodBase method)
            {
                result.Add(key + " FLAGS " + method.Attributes);
                foreach (var parameter in method.GetParameters())
                    result.Add($"{key} PARAMETER {parameter.Position} {parameter.Name} {parameter.Attributes} {parameter.ParameterType} {parameter.HasDefaultValue} {parameter.RawDefaultValue}");
            }
            if (member is FieldInfo field)
                result.Add($"{key} FIELD {field.Attributes} {(field.IsLiteral ? field.GetRawConstantValue() : null)}");
        }
    }
    return result.Order(StringComparer.Ordinal).ToArray();
}

'''
    anchor = 'var observations = new List<object>();'
    assert text.count(anchor) == 1
    text = text.replace(anchor, surface + anchor)
    anchor = '    var oldMethods = Inspect(oldAssembly); var newMethods = Inspect(newAssembly);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''
    var oldSurface = PublicSurface(oldAssembly); var newSurface = PublicSurface(newAssembly);
    if (!oldSurface.SequenceEqual(newSurface))
        throw new InvalidDataException(JsonSerializer.Serialize(new { name, removed_surface = oldSurface.Except(newSurface), added_surface = newSurface.Except(oldSurface) }));
''')
    text = text.replace('observations.Add(new { assembly = name,', 'observations.Add(new { public_surface_equal = true, public_surface = oldSurface, assembly = name,')
    path.write_text(text, encoding='utf8')
    for folder in [source, bridge, TOOLS]:
        for path in folder.rglob('*'):
            if path.is_file():
                files[rel(path)] = pin(path)
    files[rel(BASE / 'candidate.patch')] = pin(BASE / 'candidate.patch')
    save(BASE / 'source-prepared.json', dict(passed=True, files=files))
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    suites = []

    def run(name, command, is_test=False, inference=False, output=None):
        monitor.worker(state, BASE / 'processes.json', name, command, source, [0], 10 if is_test or inference else 8,
            8, 900, not inference, output)
        print(name, 'passed', flush=True)

    def restore_build(name, project, feed):
        run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', feed, '--packages', BASE / 'nuget-cache'])
        run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])

    try:
        projects = {name: source / path for name, path in [('cli', 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
            ('backend', 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
            ('tensors', 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]}
        for name, project in projects.items():
            restore_build(name, project, FEED)
        shutil.copytree(projects['cli'].parent / 'bin/Release/net10.0', runtime)
        for name, project in projects.items():
            for assembly in ['Lokad.Onnx.dll'] + ([] if name == 'tensors' else ['Lokad.Onnx.Data.dll']):
                assert pin(project.parent / 'bin/Release/net10.0' / assembly) == pin(runtime / assembly)
        reference = projects['backend'].parent / 'bin/Release/net10.0/dense-reference'
        reference.mkdir()
        shutil.copy2(DENSE, reference / 'Lokad.Onnx.Data.dll')
        restore_build('bridge', bridge / 'Bridge.csproj', FEED)
        run('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', APPLICATION / 'application-runtime', runtime,
            BASE / 'instructions.json'], output=BASE / 'bridge')
        assert read(BASE / 'instructions.json')['passed']
        assert (BASE / 'instructions.json').stat().st_size < 1024**3
        for name, pattern, disabled, passed, skipped in [
            ('focused', 'FullyQualifiedName~Lstm|FullyQualifiedName~WeSpeakerAudioTests|FullyQualifiedName~SparseMelTests', False, 203, 0),
            ('hardware-disabled', 'FullyQualifiedName~LstmOutputLane|FullyQualifiedName~LstmPanel|FullyQualifiedName~WeSpeakerAudioTests|FullyQualifiedName~SparseMelTests', True, 109, 0),
            ('backend-full', None, False, 3290, 93), ('tensors-full', None, False, 342, 0)]:
            original = monitor.clean_env
            if disabled:
                def clean_env():
                    env = original()
                    env['DOTNET_EnableHWIntrinsic'] = '0'
                    return env
                monitor.clean_env = clean_env
            try:
                project = projects['tensors' if name == 'tensors-full' else 'backend']
                command = ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                    '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results']
                if pattern:
                    command += ['--filter', pattern]
                run(name, command, is_test=True, output=BASE / 'test-results')
            finally:
                monitor.clean_env = original
            suites.append(suite(name, passed, skipped))
            save(BASE / 'suites.json', suites)
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
        shutil.copy2(TOOLS / 'PackageProbe.cs', consumer / 'Program.cs')
        run('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', BASE / 'nuget', '--source', FEED,
            '--packages', BASE / 'consumer-cache'])
        run('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        assert pin(consumer / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(runtime / 'Lokad.Onnx.dll')
        run('consumer', ['dotnet', consumer / 'bin/Release/net10.0/PackageProbe.dll',
            source / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx', pin(runtime / 'Lokad.Onnx.dll')['sha256'],
            BASE / 'consumer.json'], inference=True, output=consumer)
        assert read(BASE / 'consumer.json')['passed']
        verify(files)
        for folder in [source, runtime, bridge, consumer, BASE / 'nuget', TOOLS]:
            for path in folder.rglob('*'):
                if path.is_file() and 'obj' not in path.relative_to(folder).parts:
                    files[rel(path)] = pin(path)
        state['code'] = 0
        save(BASE / 'prepared.json', dict(passed=True, files=files, suites=suites,
            core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'), package=pin(package),
            scope='Normal source build, exact method/public surface comparison, complete suites and local package consumption; no new model timing or target promotion.'))
        print(dict(prepared=pin(BASE / 'prepared.json')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
