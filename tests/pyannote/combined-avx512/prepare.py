"""Compose the two qualified convolution paths and inventory normal builds."""
import difflib
import shutil
import traceback
from common import *


def main():
    assert not BASE.exists()
    files = {}
    for path, sha in RECEIPTS:
        assert pin(path)['sha256'] == sha
        proof = read(path)
        assert proof['passed']
        verify(proof['files'])
        for identity in proof['identities']:
            terminal(identity)
        files[rel(path)] = pin(path)
    primary = ROOT / 'artifacts/pyannote-amd-execution-v4-20260921/closed.json'
    assert pin(primary)['sha256'] == 'e0f1123a85c2a575e062fbf0e4dacc8c742c087661c2599d0197e5ca35cecc68'
    assert read(primary)['passed']
    verify(read(primary)['files'])
    files[rel(primary)] = pin(primary)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source, runtime, bridge = BASE / 'source', BASE / 'runtime', BASE / 'bridge'
    shutil.copytree(SOURCE / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    path = source / 'src/Lokad.Onnx/TensorOps.ConvPool.cs'
    before = path.read_text(encoding='utf8')
    old = '        int scratchLength = checked(blockPatch + blockOut);'
    new = old + '''
        int blockPack = ConvPackedScratchLength(M / group, C * kH * kW / group, blockN, scratchLength, options);
        int packOffset = scratchLength;
        scratchLength = checked(scratchLength + blockPack);'''
    assert before.count(old) == 1
    after = before.replace(old, new)
    old = 'RunTiledBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, b,'
    new = 'RunTiledBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, new Memory<float>(scratch, packOffset, blockPack), b,'
    assert after.count(old) == 2
    after = after.replace(old, new)
    old = 'static void RunTiledBatchFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, float[] scratch, int b,'
    new = old.replace('float[] scratch, int b,', 'float[] scratch, Memory<float> packMem, int b,')
    assert after.count(old) == 1
    after = after.replace(old, new)
    start = after.index('                var wView =', after.index('    static void RunTiledBatchFloat('))
    end = after.index('                int outBase', start)
    fallback = after[start:end]
    assert fallback.count('TryConvPortableRows') == 1 and fallback.count('MatMul2D') == 1
    dispatch = '''                if (!TryConvPackedTile(wMem.Span.Slice(g * tileM * tileKg, tileM * tileKg),
                    patchMem.Span.Slice(g * tileKg * colCount, tileKg * colCount),
                    outMem.Span.Slice(g * tileM * colCount, tileM * colCount), packMem.Span,
                    tileM, tileKg, colCount, options))
                {
''' + ''.join('    ' + line if line.strip() else line for line in fallback.splitlines(True)) + '                }\n'
    after = after[:start] + dispatch + after[end:]
    path.write_text(after, encoding='utf8')
    primary_source = FEED.parent / 'source'
    additions = {
        'src/Lokad.Onnx/Zzz.ZConvPackedRows.cs': 'src/Lokad.Onnx/TensorOps.ConvPackedRows.cs',
        'tests/Lokad.Onnx.Backend.Tests/ConvPackedRowsTests.cs': 'tests/Lokad.Onnx.Backend.Tests/ConvPackedRowsTests.cs',
    }
    for name, original in additions.items():
        target = source / name
        assert not target.exists()
        shutil.copy2(primary_source / original, target)
        files[rel(primary_source / original)] = pin(primary_source / original)
    changed, patch = [], []
    for p in sorted(source.rglob('*')):
        if not p.is_file():
            continue
        name = p.relative_to(source).as_posix()
        original = SOURCE / 'source' / name
        if not original.exists() or pin(p) != pin(original):
            changed.append(name)
            old_text = original.read_text(encoding='utf8') if original.exists() else ''
            patch.extend(difflib.unified_diff(old_text.splitlines(True), p.read_text(encoding='utf8').splitlines(True),
                fromfile='a/' + name if original.exists() else '/dev/null', tofile='b/' + name))
        files[rel(p)] = pin(p)
        if original.exists():
            files[rel(original)] = pin(original)
    assert set(changed) == {'src/Lokad.Onnx/TensorOps.ConvPool.cs', *additions}
    (BASE / 'candidate.patch').write_text(''.join(patch), encoding='utf8')
    shutil.copytree(BUILD / 'bridge', bridge, ignore=shutil.ignore_patterns('bin', 'obj'))
    shutil.copy2(TOOLS / 'Inventory.cs.txt', bridge / 'Program.cs')
    for p in [*TOOLS.iterdir(), *bridge.iterdir(), MONITOR, BASE / 'candidate.patch',
        ROOT / 'tests/pyannote/portable-integration-tests/common.py']:
        if p.is_file():
            files[rel(p)] = pin(p)
    save(BASE / 'source-prepared.json', dict(passed=True, files=files, changed=changed,
        normal_project_references=True, dispatch='AVX-512, then original portable/generic fallback'))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']

    def run(name, args, output):
        monitor.worker(state, BASE / 'processes.json', name, args, source, [0], 8, 8, 900, True, output)
        print(name, 'passed', flush=True)

    try:
        projects = {name: source / path for name, path in [('cli', 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
            ('backend', 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
            ('tensors', 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]}
        projects['bridge'] = bridge / 'Bridge.csproj'
        for name, project in projects.items():
            run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], None)
            run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], None)
        shutil.copytree(projects['cli'].parent / 'bin/Release/net10.0', runtime)
        for name in ['cli', 'backend', 'tensors']:
            for assembly in ['Lokad.Onnx.dll'] + ([] if name == 'tensors' else ['Lokad.Onnx.Data.dll']):
                assert pin(projects[name].parent / 'bin/Release/net10.0' / assembly) == pin(runtime / assembly)
        run('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', PRIOR / 'runtime', runtime,
            BASE / 'instructions.json'], bridge)
        assert read(BASE / 'instructions.json')['inventory_complete']
        verify(files)
        for folder in [source, runtime, bridge]:
            for p in folder.rglob('*'):
                if p.is_file() and 'obj' not in p.relative_to(folder).parts:
                    files[rel(p)] = pin(p)
        save(BASE / 'prepared.json', dict(passed=True, files=files, changed=changed,
            core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
            scope='Normal composition build and exhaustive inventory; instruction review and all tests pending.'))
        state['code'] = 0
        print(dict(prepared=pin(BASE / 'prepared.json'), core=pin(runtime / 'Lokad.Onnx.dll'),
            data=pin(runtime / 'Lokad.Onnx.Data.dll')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
