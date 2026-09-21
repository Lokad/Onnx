import difflib
import json
import shutil
import subprocess
import traceback
import zipfile
from common import *


def main():
    closure = read(PREVIOUS / 'closed.json')
    assert pin(PREVIOUS / 'closed.json')['sha256'] == 'cea91b423bfbb8da0059617dceeebb3f4175256480576abc925b9bdec5324629'
    assert closure['passed'] and closure['selected_for_full_model'] == 256
    verify(closure['files'])
    for identity in closure['identities']:
        terminal(identity)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source = BASE / 'source'
    source.mkdir()
    commit = subprocess.check_output(['git', 'rev-parse', 'f8c02927'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'archive', '--format=zip', '--output=' + str(BASE / 'source.zip'), commit], cwd=ROOT, check=True)
    with zipfile.ZipFile(BASE / 'source.zip') as archive:
        assert all((source / n).resolve().is_relative_to(source) for n in archive.namelist())
        archive.extractall(source)
    path = source / 'src/Lokad.Onnx/MathOps.cs'
    before = path.read_text(encoding='utf8')
    after = before
    for rows in (2, 3):
        old = f'        if (M % {rows} != 0)\n            throw new ArgumentException(nameof(M));\n\n'
        # Other two-row kernels share this guard: replace only in selected method.
        method = 'mm_unsafe_vectorized_intrinsics_2x4packed_bump' if rows == 2 else 'mm_unsafe_vectorized_intrinsics_3x4packed'
        start = after.index('public unsafe static void ' + method + '(')
        offset = after.index(old, start)
        assert offset - start < 450
        replacement = old + '        if (TryPackedPartialSums(M, N, K, A, P, C)) return;\n\n'
        after = after[:offset] + after[offset:].replace(old, replacement, 1)
    path.write_text(after, encoding='utf8')
    shutil.copy2(TOOLS / 'MathOps.PartialReduction.cs', source / 'src/Lokad.Onnx/MathOps.PartialReduction.cs')
    (BASE / 'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/src/Lokad.Onnx/MathOps.cs', tofile='b/src/Lokad.Onnx/MathOps.cs')), encoding='utf8')
    bridge = BASE / 'bridge'
    bridge.mkdir()
    bridge_source = ROOT / 'tests/whisper/memory-product-v2/CompareIlStable.cs'
    code = bridge_source.read_text(encoding='utf8')
    old = '    if (oldMethods.Count == 0 || !oldMethods.Keys.Order().SequenceEqual(newMethods.Keys.Order())) throw new InvalidDataException("Method coverage differs");'
    new = '''    var added = newMethods.Keys.Except(oldMethods.Keys).ToArray();
    if (oldMethods.Count == 0 || oldMethods.Keys.Except(newMethods.Keys).Any()
        || (name == "Lokad.Onnx.dll" ? added.Length != 1 || !added[0].StartsWith("Lokad.Onnx.MathOps::TryPackedPartialSums::") : added.Length != 0))
        throw new InvalidDataException("Unexpected added/removed method");'''
    assert code.count(old) == 1
    code = code.replace(old, new)
    old = '    if (differences.Length != 0) throw new InvalidDataException("Runtime IL differs: " + string.Join("; ", differences));'
    new = '''    string[] expected = name == "Lokad.Onnx.dll" ? ["mm_unsafe_vectorized_intrinsics_2x4packed_bump", "mm_unsafe_vectorized_intrinsics_3x4packed"] : [];
    if (!differences.Select(k => k.Split("::")[1]).Order().SequenceEqual(expected.Order()))
        throw new InvalidDataException("Unexpected changed methods: " + string.Join("; ", differences));'''
    assert code.count(old) == 1
    code = code.replace(old, new)
    old = '        normalized_methods = oldMethods, equal = true });'
    assert code.count(old) == 1
    code = code.replace(old, '''        normalized_methods = oldMethods, unchanged_methods = oldMethods.Count - differences.Length,
        changed_methods = differences.ToDictionary(k => k, k => newMethods[k]),
        added_methods = added.ToDictionary(k => k, k => newMethods[k]), equal_except_partial_sums = true });''')
    code = code.replace('All Core/Data method IL, resolved operands, locals, stack and exception clauses', 'Only two packed dispatches and one partial-sum helper differ')
    (bridge / 'Program.cs').write_text(code, encoding='utf8')
    (bridge / 'Bridge.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>\n')
    probe = BASE / 'probe'
    probe.mkdir()
    for name in ('Probe.cs', 'Probe.csproj'):
        shutil.copy2(TOOLS / name, probe / name)
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    feed = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
    runtime = BASE / 'runtime'
    def command(label, args, cwd=ROOT):
        return monitor.worker(state, BASE / 'build-state.json', label, args, cwd, [0], 4, 2, 600, True, BASE / 'source/src/Lokad.Onnx/bin')
    def build(label, project, extra):
        command(label + '-restore', ['dotnet', 'restore', project, *flags, '--source', feed, '--packages', BASE / 'packages', *extra])
        command(label + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', *extra])
    try:
        build('core', source / 'src/Lokad.Onnx/Lokad.Onnx.csproj', [])
        shutil.copytree(ORIGINAL, runtime)
        shutil.copy2(source / 'src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll', runtime / 'Lokad.Onnx.dll')
        build('bridge', bridge / 'Bridge.csproj', [])
        command('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', ORIGINAL, runtime, BASE / 'instructions.json'])
        build('probe', probe / 'Probe.csproj', ['-p:FrozenProductDirectory=' + str(runtime)])
        probe_bin = probe / 'bin/Release/net10.0/Probe.dll'
        command('geometry', ['dotnet', probe_bin, ROOT, ORIGINAL, BASE / 'geometry.json', 'normal'])
        old_env = monitor.clean_env
        monitor.clean_env = lambda: dict(old_env(), DOTNET_EnableHWIntrinsic='0')
        try:
            command('hardware-off', ['dotnet', probe_bin, ROOT, ORIGINAL, BASE / 'hardware-off.json', 'disabled'])
        finally:
            monitor.clean_env = old_env
        for p in (NATIVE / 'replay').iterdir():
            if p.is_file() and p.name.startswith('TranscribeReplay.'):
                shutil.copy2(p, runtime / p.name)
        for p in (ROOT / 'artifacts/audio-ort-baseline-v2-20260919/bin').iterdir():
            if p.is_file() and p.name.startswith('AudioBenchmark.'):
                shutil.copy2(p, runtime / p.name)
        assert pin(runtime / 'TranscribeReplay.dll')['sha256'] == '335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
        assert pin(runtime / 'AudioBenchmark.dll')['sha256'] == '7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
        assert pin(runtime / 'Lokad.Onnx.Data.dll') == pin(ORIGINAL / 'Lokad.Onnx.Data.dll')
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'build-state.json', state)
    files = {}
    for p in BASE.rglob('*'):
        relative = p.relative_to(BASE)
        if p.is_file() and not {'obj', 'packages'}.intersection(relative.parts):
            files[rel(p)] = pin(p)
    for p in [*TOOLS.iterdir(), PREVIOUS / 'closed.json', MONITOR, bridge_source, REFERENCE, CORPUS, BASELINE,
              ROOT / 'tests/parakeet/transcribe/audit.py', ROOT / 'tests/audio/comparison/audit.py']:
        if p.is_file():
            files[rel(p)] = pin(p)
    for name, wanted in read(REFERENCE)['files'].items():
        p = REFERENCE.parent / name
        assert pin(p) == {k: wanted[k] for k in ('bytes', 'sha256')}
        files[rel(p)] = pin(p)
    manifest = read(CORPUS)
    for value in [*manifest['models'].values(), manifest['reference'], manifest['upstream'], *[c['pcm'] for c in manifest['cases']]]:
        assert pin(ROOT / value['path']) == {k: value[k] for k in ('bytes', 'sha256')}
        files[value['path']] = pin(ROOT / value['path'])
    for p in Path(str(BASELINE) + '.tensors').iterdir():
        if p.is_file():
            files[rel(p)] = pin(p)
    save(BASE / 'prepared.json', dict(passed=True, source=commit, files=files,
         core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
         native=pin(runtime / 'TranscribeReplay.dll'), public=pin(runtime / 'AudioBenchmark.dll'),
         scope='Two packed dispatch changes and one helper; numerical application qualification pending'))
    print(json.dumps(dict(passed=True, core=pin(runtime / 'Lokad.Onnx.dll'), prepared=pin(BASE / 'prepared.json'))))


if __name__ == '__main__':
    main()
