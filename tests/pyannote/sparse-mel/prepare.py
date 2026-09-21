"""Build only sparse-mel Data, prove method isolation and retain the exact dense reference."""
from common import *
import difflib
import xml.etree.ElementTree as ET


def main():
    assert not BASE.exists()
    receipts = [
        (PRIOR / 'closed.json', '27ff741eb0a6aceb354806a030730175a95e429d24ae5c307a31a7633b756669'),
        (QUALIFIED / 'closed.json', '20fd9f38b80f2f1e2b87a375b0b2c1c99e255d77f69b674271614361e7282002'),
        (COMPARISON / 'closed.json', 'f159d73999c5f7f16b4bfe1e180204597bd4f321aa590179be753b417109db3d'),
        (RELEASE / 'closed.json', '5b7bef7bc9fb648062438a0014a2afedd5dfe359697c49484685208e271c4210')]
    for receipt, sha in receipts:
        assert pin(receipt)['sha256'] == sha
        closed = read(receipt)
        assert closed['passed']
        verify(closed['files'])
        for identity in closed.get('identities', closed.get('terminal_identities', [])):
            terminal(identity)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source, runtime, bridge = BASE / 'source', BASE / 'runtime', BASE / 'bridge'
    shutil.copytree(PRIOR / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    shutil.copytree(QUALIFIED / 'application-runtime', runtime)
    path = source / 'src/Lokad.Onnx.Data/WeSpeakerAudio.cs'
    before = path.read_text(encoding='utf8')
    old = '    static readonly float[] MelWeights = CreateMelWeights();'
    assert before.count(old) == 1
    after = before.replace(old, old + '\n    static readonly (int Start, int End)[] MelSupport = CreateMelSupport();')
    old = '                for (int k = 0; k < FourierSize / 2; k++) energy += (double)powers[k] * MelWeights[mel * (FourierSize / 2) + k];'
    assert after.count(old) == 1
    after = after.replace(old, '''                var (first, last) = MelSupport[mel];
                for (int k = first; k < last; k++) energy += (double)powers[k] * MelWeights[mel * (FourierSize / 2) + k];''')
    anchor = '    static Complex[] CreateRoots()'
    assert after.count(anchor) == 1
    after = after.replace(anchor, '''    // Keep the original coefficient values and ascending accumulation order.
    // Valid normalized PCM keeps every power finite, so omitted zero terms
    // cannot change a nonnegative energy or its signed-zero behavior.
    static (int Start, int End)[] CreateMelSupport()
    {
        var support = new (int Start, int End)[MelBins];
        const int bins = FourierSize / 2;
        for (int band = 0; band < MelBins; band++)
        {
            int first = 0, last = bins;
            while (first < last && MelWeights[band * bins + first] == 0f) first++;
            while (last > first && MelWeights[band * bins + last - 1] == 0f) last--;
            support[band] = (first, last);
        }
        return support;
    }

''' + anchor)
    path.write_text(after, encoding='utf8')
    (BASE / 'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/src/Lokad.Onnx.Data/WeSpeakerAudio.cs', tofile='b/src/Lokad.Onnx.Data/WeSpeakerAudio.cs')), encoding='utf8')
    project = source / 'src/Lokad.Onnx.Data/Lokad.Onnx.Data.csproj'
    text = project.read_text(encoding='utf8')
    old_runtime = str(ROOT / 'artifacts/pyannote-request-contexts-v3-20260921/runtime')
    assert text.count(old_runtime) == 2
    project.write_text(text.replace(old_runtime, str(runtime)), encoding='utf8')
    shutil.copy2(TOOLS / 'SparseMelTests.cs', source / 'tests/Lokad.Onnx.Backend.Tests/SparseMelTests.cs')
    shutil.copytree(PRIOR / 'bridge', bridge, ignore=shutil.ignore_patterns('bin', 'obj'))
    path = bridge / 'Program.cs'
    text = path.read_text(encoding='utf8')
    start, end = text.index('    object? compilerRename = null;'), text.index('    var removed =')
    text = text[:start] + '    object? compilerRename = null;\n' + text[end:]
    start, end = text.index('    const string tensor ='), text.index('    if (!allowed)')
    text = text[:start] + '''    const string audio = "Lokad.Onnx.WeSpeakerAudio::";
    bool allowed = name == "Lokad.Onnx.dll"
        ? removed.Length == 0 && added.Length == 0 && differences.Length == 0
          && Hash(Path.Combine(before, name)) == Hash(Path.Combine(after, name))
        : removed.Length == 0 && added.Length == 1
          && added[0].StartsWith(audio + "CreateMelSupport::")
          && differences.Length == 2
          && differences.Count(k => k.StartsWith(audio + "LogMelFilterbank::")) == 1
          && differences.Count(k => k.StartsWith(audio + ".cctor::")) == 1;
''' + text[end:]
    path.write_text(text.replace('equal_except_portable_convolution_dispatch', 'equal_except_sparse_mel'), encoding='utf8')
    backend = source / 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
    text = backend.read_text(encoding='utf8')
    assert text.count(str(PRIOR / 'runtime')) == 6
    backend.write_text(text.replace(str(PRIOR / 'runtime'), str(runtime)), encoding='utf8')
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    state_path = BASE / 'preparation.json'
    save(state_path, state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']

    def run(name, args, preflight, output):
        monitor.worker(state, state_path, name, args, source, [0], preflight, 8, 900, True, output)
        print(name, 'passed', flush=True)

    try:
        for name, target in [('data', project), ('bridge', bridge / 'Bridge.csproj')]:
            run(name + '-restore', ['dotnet', 'restore', target, *flags, '--source', FEED, '--packages', BASE / 'packages'], 8, None)
            run(name + '-build', ['dotnet', 'build', target, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], 8, None)
        shutil.copy2(project.parent / 'bin/Release/net10.0/Lokad.Onnx.Data.dll', runtime / 'Lokad.Onnx.Data.dll')
        assert pin(runtime / 'Lokad.Onnx.dll') == pin(QUALIFIED / 'application-runtime/Lokad.Onnx.dll')
        run('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', QUALIFIED / 'application-runtime', runtime, BASE / 'instructions.json'], 8, BASE / 'bridge/bin')
        assert read(BASE / 'instructions.json')['passed']
        run('backend-restore', ['dotnet', 'restore', backend, *flags, '--source', FEED, '--packages', BASE / 'packages'], 8, None)
        run('backend-build', ['dotnet', 'build', backend, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], 8, None)
        output = backend.parent / 'bin/Release/net10.0'
        for name in ['Lokad.Onnx', 'Lokad.Onnx.Data', 'Google.Protobuf', 'FastBertTokenizer', 'Lokad.Tokenizers', 'SixLabors.ImageSharp']:
            assert pin(output / (name + '.dll')) == pin(runtime / (name + '.dll'))
        reference = output / 'dense-reference'
        reference.mkdir()
        shutil.copy2(QUALIFIED / 'application-runtime/Lokad.Onnx.Data.dll', reference / 'Lokad.Onnx.Data.dll')
        files = {rel(p): pin(p) for p in [MONITOR, *[p for p, _ in receipts], BASE / 'candidate.patch']}
        for folder in [source, runtime, bridge, TOOLS]:
            for path in folder.rglob('*'):
                if path.is_file() and 'obj' not in path.relative_to(folder).parts:
                    files[rel(path)] = pin(path)
        save(BASE / 'focused-prepared.json', dict(passed=True, files=files, core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll')))
        for name, pattern, disabled in [
            ('focused', 'FullyQualifiedName~WeSpeakerAudioTests|FullyQualifiedName~SparseMelTests', False),
            ('hardware-disabled', 'FullyQualifiedName~WeSpeakerAudioTests|FullyQualifiedName~SparseMelTests', True)]:
            original = monitor.clean_env
            if disabled:
                def environment():
                    env = original()
                    env['DOTNET_EnableHWIntrinsic'] = '0'
                    return env
                monitor.clean_env = environment
            try:
                run(name, ['dotnet', 'test', backend, '-c', 'Release', *flags, '--no-build', '--no-restore', '--filter', pattern,
                    '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results'], 10, BASE / 'test-results')
            finally:
                monitor.clean_env = original
            tree = ET.parse(BASE / 'test-results' / (name + '.trx'))
            counters = tree.find('.//{*}Counters').attrib
            assert int(counters['failed']) == 0 and int(counters['passed']) > 58
            new_tests = [r for r in tree.findall('.//{*}UnitTestResult') if 'SparseMelTests.' in r.attrib['testName']]
            assert len(new_tests) == 58 and all(r.attrib['outcome'] == 'Passed' for r in new_tests)
            save(BASE / (name + '.json'), dict(passed=True, counters=counters, sparse_cases=len(new_tests)))
        verify(files)
        state['code'] = 0
        save(BASE / 'prepared.json', dict(passed=True, files=files, core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
            scope='Sparse-mel Data method and focused frontend qualification; real-audio/public/performance checks pending.'))
        print(dict(prepared=pin(BASE / 'prepared.json'), data=pin(runtime / 'Lokad.Onnx.Data.dll')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(state_path, state)


if __name__ == '__main__':
    main()
