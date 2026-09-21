"""Build a separate Core and test poisoned output reuse with exact Data."""
import json
import re
import shutil
import traceback
import xml.etree.ElementTree as ET
from common import *
from modify_source import modify


def main():
    assert not BASE.exists()
    receipt = PRIOR / 'closed.json'
    assert pin(receipt)['sha256'] == 'fb7d8a4df2a7e6f51896486d1f575605f40936ebc9d3d8b373aaf2840495d10a'
    closed = read(receipt)
    assert closed['passed']
    verify(closed['files'])
    for identity in closed['identities']:
        terminal(identity)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source = BASE / 'source'
    shutil.copytree(PRIOR / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    runtime = BASE / 'runtime'
    shutil.copytree(PRIOR / 'runtime', runtime)
    patch, changes = modify(source)
    (BASE / 'candidate.patch').write_text(patch, encoding='utf8')
    save(BASE / 'source-changes.json', dict(files=[c['path'] for c in changes],
        scope='Pool dispatch/required overloads/cleared float output allocation; unchanged kernel bodies and geometry.'))
    shutil.copy2(TOOLS / 'ConvolutionPoolTests.cs', source / 'tests/Lokad.Onnx.Backend.Tests/ConvolutionPoolTests.cs')
    # Resolve normalized operands using the already-qualified IL inspector.
    bridge = BASE / 'bridge'
    shutil.copytree(PRIOR / 'bridge', bridge, ignore=shutil.ignore_patterns('bin', 'obj'))
    path = bridge / 'Program.cs'
    text = path.read_text(encoding='utf8')
    first = text.index('    const string pipeline = ')
    last = text.index('    if (!allowed)', first)
    text = text[:first] + '''    const string tensor = "Lokad.Onnx.Tensor`1[T]::";
    bool allowed = name == "Lokad.Onnx.Data.dll"
        ? removed.Length == 0 && added.Length == 0 && differences.Length == 0
          && Hash(Path.Combine(before, name)) == Hash(Path.Combine(after, name))
        : removed.Length == 1 && removed[0].StartsWith(tensor + "Conv2DFloatCore::")
          && added.Length == 5 && added.All(k => k.Contains("TensorBufferPool"))
          && added.Count(k => k.StartsWith(tensor + "Conv2DFloatCore::")) == 1
          && added.Count(k => k.StartsWith(tensor + "Conv2D::")) == 2
          && added.Count(k => k.StartsWith("Lokad.Onnx.CPUExecutionProvider::Conv::")) == 1
          && added.Count(k => k.StartsWith("Lokad.Onnx.CPUExecutionProvider::ConvRelu::")) == 1
          && differences.Length == 5
          && differences.Count(k => k.StartsWith(tensor + "Conv2D::")) == 2
          && differences.Count(k => k.StartsWith("Lokad.Onnx.CPUExecutionProvider::Conv::")) == 1
          && differences.Count(k => k.StartsWith("Lokad.Onnx.CPUExecutionProvider::ConvRelu::")) == 1
          && differences.Count(k => k.StartsWith("Lokad.Onnx.Node::ExecuteCPU::")) == 1;
''' + text[last:]
    text = text.replace('equal_except_request_lifetime = true', 'equal_except_convolution_output_allocation = true')
    path.write_text(text, encoding='utf8')
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    save(BASE / 'preparation.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']

    def command(name, args, preflight, output):
        row = monitor.worker(state, BASE / 'preparation.json', name, args, source, [0], preflight, 8, 900, True, output)
        print(name, 'passed', flush=True)
        return row

    try:
        for name, project in [('core', source / 'src/Lokad.Onnx/Lokad.Onnx.csproj'), ('bridge', bridge / 'Bridge.csproj')]:
            command(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], 8, None)
            command(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], 8, None)
        shutil.copy2(source / 'src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll', runtime / 'Lokad.Onnx.dll')
        assert pin(runtime / 'Lokad.Onnx.Data.dll') == pin(PRIOR / 'runtime/Lokad.Onnx.Data.dll')
        command('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', PRIOR / 'runtime', runtime, BASE / 'instructions.json'], 8, BASE / 'bridge/bin')
        assert read(BASE / 'instructions.json')['passed']
        project = source / 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
        before = project.read_text(encoding='utf8')
        after, count = re.subn(r'    <ProjectReference Include="[^"]+" />', '', before)
        assert count == 2
        dependencies = ['Lokad.Onnx', 'Lokad.Onnx.Data', 'Google.Protobuf', 'FastBertTokenizer', 'Lokad.Tokenizers', 'SixLabors.ImageSharp']
        references = '\n'.join(f'    <Reference Include="{n}"><HintPath>{runtime / (n + ".dll")}</HintPath></Reference>' for n in dependencies)
        project.write_text(after.replace('</Project>', '  <ItemGroup>\n' + references + '\n  </ItemGroup>\n</Project>'), encoding='utf8')
        command('backend-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], 8, None)
        command('backend-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], 8, None)
        for name in dependencies:
            assert pin(project.parent / 'bin/Release/net10.0' / (name + '.dll')) == pin(runtime / (name + '.dll'))
        files = {rel(receipt): pin(receipt), rel(MONITOR): pin(MONITOR)}
        for folder in [source, runtime, bridge, TOOLS]:
            for path in folder.rglob('*'):
                if path.is_file() and 'obj' not in path.relative_to(folder).parts:
                    files[rel(path)] = pin(path)
        save(BASE / 'focused-prepared.json', dict(passed=True, files=files, limits=dict(preflight_gib=10, rss_gib=8, seconds=900)))
        for name, filter, scalar in [
            ('focused', 'FullyQualifiedName~ConvolutionPoolTests|FullyQualifiedName~Conv|FullyQualifiedName~PoolLifetime|FullyQualifiedName~GraphOwnership', False),
            ('hardware-disabled', 'FullyQualifiedName~ConvolutionPoolTests', True)]:
            original = monitor.clean_env
            if scalar:
                def fallback_env():
                    env = original()
                    env['DOTNET_EnableHWIntrinsic'] = '0'
                    return env
                monitor.clean_env = fallback_env
            try:
                args = ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore', '--filter', filter,
                    '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results']
                command(name, args, 10, BASE / 'test-results')
            finally:
                monitor.clean_env = original
            counters = ET.parse(BASE / 'test-results' / (name + '.trx')).find('.//{*}Counters').attrib
            assert int(counters['failed']) == 0 and int(counters['passed']) >= 31
            save(BASE / (name + '.json'), dict(passed=True, counters=counters))
        verify(files)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'preparation.json', state)
    files[rel(BASE / 'candidate.patch')] = pin(BASE / 'candidate.patch')
    save(BASE / 'prepared.json', dict(passed=True, files=files,
        core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
        scope='Focused output-pool correctness and method isolation; complete graph/application qualification pending.'))
    print(json.dumps(dict(passed=True, prepared=pin(BASE / 'prepared.json'), core=pin(runtime / 'Lokad.Onnx.dll'))))


if __name__ == '__main__':
    main()
