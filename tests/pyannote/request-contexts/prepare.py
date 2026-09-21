"""Build a Data-only request-lifetime candidate over the exact qualified Core."""
import difflib
import json
import shutil
import traceback
from common import *


def main():
    assert not BASE.exists()
    receipt = CANDIDATE / 'qualification-closed.json'
    assert pin(receipt)['sha256'] == 'cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53'
    qualified = read(receipt); assert qualified['passed']; verify(qualified['files'])
    assert pin(PROBE / 'closed.json')['sha256'] == 'd06bffe6d4aba50e7b17373c2a155234c514419bd4859063938fb1fe436fb626'
    verify(read(PROBE / 'closed.json')['files']); assert read(PROBE / 'analysis.json')['request_scoped_prototype_admitted']
    BASE.mkdir(); (BASE / 'logs').mkdir()
    source = BASE / 'source'; shutil.copytree(CANDIDATE / 'candidate-source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    runtime = BASE / 'runtime'; shutil.copytree(CANDIDATE / 'runtimes/candidate', runtime)
    changes = []
    def edit(path, before, after, reason):
        assert before != after
        path.write_text(after, encoding='utf8')
        changes.append(dict(path=path.relative_to(source).as_posix(), reason=reason,
            diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/'+path.relative_to(source).as_posix(), tofile='b/'+path.relative_to(source).as_posix()))))
    path = source / 'src/Lokad.Onnx.Data/WeSpeakerEmbedder.cs'; before = path.read_text(encoding='utf8')
    start = before.index('    internal WeSpeakerEmbedding[] ExtractPipeline(')
    prefix, method = before[:start], before[start:]
    old = 'float[] samples, float[][] masks, CancellationToken cancellation)'
    assert method.count(old) == 1
    method = method.replace(old, 'float[] samples, float[][] masks, CancellationToken cancellation, PipelineRequest? request = null)')
    old = '            cancellation.ThrowIfCancellationRequested();\n            var features = WeSpeakerAudio.LogMelFilterbank(samples, 16000, cancellation);'
    assert method.count(old) == 1
    method = method.replace(old, '            if (request is not null) ObjectDisposedException.ThrowIf(request.Disposed, request);\n' + old)
    old = '            var encoding = encoder.CreateExecution(ExecutionOptions.Memory);\n            var projecting = projection.CreateExecution(ExecutionOptions.Memory);'
    assert method.count(old) == 1
    method = method.replace(old, '''            var encoding = request is null ? encoder.CreateExecution(ExecutionOptions.Memory)
                : (request.Encoding ??= encoder.CreateExecution(ExecutionOptions.Memory));
            var projecting = request is null ? projection.CreateExecution(ExecutionOptions.Memory)
                : (request.Projecting ??= projection.CreateExecution(ExecutionOptions.Memory));''')
    after = prefix + method; end = after.rindex('\n}')
    after = after[:end] + '\n' + (TOOLS / 'PipelineRequest.cs.txt').read_text() + after[end:]
    edit(path, before, after, 'Request-owned lazy encoder/projection contexts; unchanged public standalone extraction')
    path = source / 'src/Lokad.Onnx.Data/Community1Diarizer.cs'; before = path.read_text(encoding='utf8')
    first = before.index('            for (int c = 0; c < chunks; c++)', before.index('var embeddings ='))
    last = before.index('            var clustered =', first)
    loop = before[first:last]
    assert loop.count('embedder.ExtractPipeline(') == 1
    loop = loop.replace('embedder.ExtractPipeline(', 'request.ExtractPipeline(')
    after = before[:first] + '            using (var request = embedder.CreatePipelineRequest())\n            {\n' + ''.join('    '+line if line.strip() else line for line in loop.splitlines(True)) + '            }\n' + before[last:]
    edit(path, before, after, 'Reuse within the embedding loop; release contexts before clustering on every exit')
    product_changes = list(changes)
    (BASE / 'candidate.patch').write_text(''.join(c['diff'] for c in product_changes), encoding='utf8')
    path = source / 'src/Lokad.Onnx.Data/Lokad.Onnx.Data.csproj'; before = path.read_text(encoding='utf8')
    old = '    <ProjectReference Include="..\\Lokad.Onnx\\Lokad.Onnx.csproj" />'
    assert before.count(old) == 1
    after = before.replace(old, f'    <Reference Include="Lokad.Onnx"><HintPath>{runtime / "Lokad.Onnx.dll"}</HintPath></Reference>\n    <Reference Include="Google.Protobuf"><HintPath>{runtime / "Google.Protobuf.dll"}</HintPath></Reference>')
    edit(path, before, after, 'Build only Data against frozen Core')
    shutil.copy2(TOOLS / 'PipelineRequestTests.cs', source / 'tests/Lokad.Onnx.Backend.Tests/PipelineRequestTests.cs')
    bridge = BASE / 'bridge'; bridge.mkdir()
    original = ROOT / 'tests/whisper/memory-product-v2/CompareIlStable.cs'
    code = original.read_text(encoding='utf8')
    first = code.index('    if (oldMethods.Count == 0 ||')
    last = code.index('    bool? correctedAnnotation', first)
    code = code[:first] + '''    var removed = oldMethods.Keys.Except(newMethods.Keys).ToArray();
    var added = newMethods.Keys.Except(oldMethods.Keys).ToArray();
    var differences = oldMethods.Keys.Intersect(newMethods.Keys).Where(k => oldMethods[k] != newMethods[k]).ToArray();
    const string pipeline = "Lokad.Onnx.WeSpeakerEmbedder::ExtractPipeline::";
    bool allowed = name == "Lokad.Onnx.dll"
        ? removed.Length == 0 && added.Length == 0 && differences.Length == 0 && Hash(Path.Combine(before, name)) == Hash(Path.Combine(after, name))
        : removed.Length == 1 && removed[0].StartsWith(pipeline) && added.Length == 5
          && added.Count(k => k.StartsWith(pipeline)) == 1
          && added.Count(k => k.StartsWith("Lokad.Onnx.WeSpeakerEmbedder::CreatePipelineRequest::")) == 1
          && added.Count(k => k.StartsWith("Lokad.Onnx.WeSpeakerEmbedder+PipelineRequest::")) == 3
          && differences.Length == 1 && differences[0].StartsWith("Lokad.Onnx.Community1Diarizer::Diarize::");
    if (!allowed) throw new InvalidDataException(JsonSerializer.Serialize(new { name, removed, added, differences }));
''' + code[last:]
    old = '        normalized_methods = oldMethods, equal = true });'
    assert code.count(old) == 1
    code = code.replace(old, '        normalized_methods = oldMethods, unchanged_methods = oldMethods.Count - removed.Length - differences.Length, removed, added, differences,\n        candidate_methods = newMethods.Where(p => added.Contains(p.Key) || differences.Contains(p.Key)).ToDictionary(p => p.Key, p => p.Value), equal_except_request_lifetime = true });')
    (bridge / 'Program.cs').write_text(code, encoding='utf8')
    (bridge / 'Bridge.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>\n', encoding='utf8')
    save(BASE / 'source-changes.json', changes)
    own = psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    def command(name, args, cwd): return monitor.worker(state, BASE / 'builds.json', name, args, cwd, [0], 8, 8, 900, True, None)
    try:
        for name, project in [('data', path), ('bridge', bridge / 'Bridge.csproj')]:
            command(name+'-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], source)
            command(name+'-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], source)
        shutil.copy2(source / 'src/Lokad.Onnx.Data/bin/Release/net10.0/Lokad.Onnx.Data.dll', runtime / 'Lokad.Onnx.Data.dll')
        assert pin(runtime / 'Lokad.Onnx.dll') == pin(CANDIDATE / 'runtimes/candidate/Lokad.Onnx.dll')
        command('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', CANDIDATE / 'runtimes/candidate', runtime, BASE / 'instructions.json'], ROOT)
        assert read(BASE / 'instructions.json')['passed']
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'builds.json', state)
    files = {rel(receipt): pin(receipt), rel(PROBE / 'closed.json'): pin(PROBE / 'closed.json'), rel(MONITOR): pin(MONITOR), rel(original): pin(original)}
    for folder in [BASE, TOOLS]:
        for p in folder.rglob('*'):
            if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(folder).parts): files[rel(p)] = pin(p)
    save(BASE / 'prepared.json', dict(passed=True, files=files, runtime={p.name: pin(p) for p in runtime.glob('*.dll')},
        scope='Data-only request-context lifetime prototype; public/model qualification pending.'))
    print(json.dumps(dict(passed=True, core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'), prepared=pin(BASE / 'prepared.json'))))


if __name__ == '__main__': main()
