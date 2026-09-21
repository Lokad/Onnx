"""Change only tiled bias addition, retaining every other Core method and Data bytes."""
from common import *
import difflib
import xml.etree.ElementTree as ET


def main():
    assert not BASE.exists()
    receipts = [
        (PRIOR / 'closed.json', '27ff741eb0a6aceb354806a030730175a95e429d24ae5c307a31a7633b756669'),
        (QUALIFIED / 'closed.json', '20fd9f38b80f2f1e2b87a375b0b2c1c99e255d77f69b674271614361e7282002'),
        (COMPARISON / 'closed.json', 'f159d73999c5f7f16b4bfe1e180204597bd4f321aa590179be753b417109db3d'),
        (COVERAGE / 'closed.json', 'b1c0c49b02e39fac2c3aa2660419e16019cd05137036627855aa384fb340a868')]
    for receipt, sha in receipts:
        assert pin(receipt)['sha256'] == sha
        closed = read(receipt)
        assert closed['passed']
        verify(closed['files'])
        for identity in closed.get('identities', closed.get('terminal_identities', [])):
            terminal(identity)
    assert read(COMPARISON / 'analysis.json')['qualifies_for_later_amd']
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source = BASE / 'source'
    shutil.copytree(PRIOR / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    runtime = BASE / 'runtime'
    shutil.copytree(QUALIFIED / 'application-runtime', runtime)
    path = source / 'src/Lokad.Onnx/TensorOps.ConvPool.cs'
    before = path.read_text(encoding='utf8')
    old = '''                    for (int j = 0; j < colCount; j++)
                    {
                        float v = hasBias ? ds[blkRow + j] + bi : ds[blkRow + j];'''
    new = '''                    int j = 0;
                    if (hasBias && options.UseSimd && Vector.IsHardwareAccelerated)
                    {
                        var biasVector = new Vector<float>(bi);
                        int width = Vector<float>.Count;
                        for (; j <= colCount - width; j += width)
                        {
                            var values = new Vector<float>(ds.Slice(blkRow + j, width));
                            (values + biasVector).CopyTo(os.Slice(outRow + j, width));
                        }
                    }
                    for (; j < colCount; j++)
                    {
                        float v = hasBias ? ds[blkRow + j] + bi : ds[blkRow + j];'''
    assert before.count(old) == 1
    after = before.replace(old, new)
    path.write_text(after, encoding='utf8')
    (BASE / 'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/src/Lokad.Onnx/TensorOps.ConvPool.cs', tofile='b/src/Lokad.Onnx/TensorOps.ConvPool.cs')), encoding='utf8')
    shutil.copy2(TOOLS / 'VectorBiasTests.cs', source / 'tests/Lokad.Onnx.Backend.Tests/VectorBiasTests.cs')
    bridge = BASE / 'bridge'
    shutil.copytree(PRIOR / 'bridge', bridge, ignore=shutil.ignore_patterns('bin', 'obj'))
    path = bridge / 'Program.cs'
    text = path.read_text(encoding='utf8')
    start = text.index('    object? compilerRename = null;')
    end = text.index('    var removed =', start)
    text = text[:start] + '    object? compilerRename = null;\n' + text[end:]
    start = text.index('        : removed.Length == 0 && added.Length == 2')
    end = text.index('    if (!allowed)', start)
    text = text[:start] + '''        : removed.Length == 0 && added.Length == 0
          && differences.Length == 1 && differences[0].StartsWith(tensor + "RunTiledBatchFloat::");
''' + text[end:]
    text = text.replace('equal_except_portable_convolution_dispatch', 'equal_except_vector_bias')
    path.write_text(text, encoding='utf8')
    project = source / 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
    text = project.read_text(encoding='utf8')
    assert text.count(str(PRIOR / 'runtime')) == 6
    project.write_text(text.replace(str(PRIOR / 'runtime'), str(runtime)), encoding='utf8')
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    state_path = BASE / 'preparation.json'
    save(state_path, state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']

    def run(name, args, preflight, output):
        monitor.worker(state, state_path, name, args, source, [0], preflight, 8, 900, True, output)
        print(name, 'passed', flush=True)

    try:
        for name, target in [('core', source / 'src/Lokad.Onnx/Lokad.Onnx.csproj'), ('bridge', bridge / 'Bridge.csproj')]:
            run(name + '-restore', ['dotnet', 'restore', target, *flags, '--source', FEED, '--packages', BASE / 'packages'], 8, None)
            run(name + '-build', ['dotnet', 'build', target, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], 8, None)
        shutil.copy2(source / 'src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll', runtime / 'Lokad.Onnx.dll')
        assert pin(runtime / 'Lokad.Onnx.Data.dll') == pin(QUALIFIED / 'application-runtime/Lokad.Onnx.Data.dll')
        run('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', QUALIFIED / 'application-runtime', runtime, BASE / 'instructions.json'], 8, BASE / 'bridge/bin')
        assert read(BASE / 'instructions.json')['passed']
        run('backend-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], 8, None)
        run('backend-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], 8, None)
        for name in ['Lokad.Onnx', 'Lokad.Onnx.Data', 'Google.Protobuf', 'FastBertTokenizer', 'Lokad.Tokenizers', 'SixLabors.ImageSharp']:
            assert pin(project.parent / 'bin/Release/net10.0' / (name + '.dll')) == pin(runtime / (name + '.dll'))
        files = {rel(p): pin(p) for p in [MONITOR, *[p for p, _ in receipts], BASE / 'candidate.patch']}
        for folder in [source, runtime, bridge, TOOLS]:
            for path in folder.rglob('*'):
                if path.is_file() and 'obj' not in path.relative_to(folder).parts:
                    files[rel(path)] = pin(path)
        save(BASE / 'focused-prepared.json', dict(passed=True, files=files, core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll')))
        for name, pattern, disabled, count in [
            ('focused', 'FullyQualifiedName~Conv|FullyQualifiedName~PoolLifetime|FullyQualifiedName~GraphOwnership|FullyQualifiedName~VectorBiasTests', False, 358),
            ('hardware-disabled', 'FullyQualifiedName~VectorBiasTests', True, 144)]:
            original = monitor.clean_env
            if disabled:
                def environment():
                    env = original()
                    env['DOTNET_EnableHWIntrinsic'] = '0'
                    return env
                monitor.clean_env = environment
            try:
                run(name, ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore', '--filter', pattern,
                    '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results'], 10, BASE / 'test-results')
            finally:
                monitor.clean_env = original
            counters = ET.parse(BASE / 'test-results' / (name + '.trx')).find('.//{*}Counters').attrib
            assert int(counters['failed']) == 0 and int(counters['passed']) == count
            save(BASE / (name + '.json'), dict(passed=True, counters=counters))
        verify(files)
        state['code'] = 0
        save(BASE / 'prepared.json', dict(passed=True, files=files, core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
            scope='One-method vector-bias focused qualification; graph/public/performance checks pending.'))
        print(dict(prepared=pin(BASE / 'prepared.json'), core=pin(runtime / 'Lokad.Onnx.dll')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(state_path, state)


if __name__ == '__main__':
    main()
