"""Defer only unused wrappers; normal build and original complete tests."""
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
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source, runtime, bridge = BASE / 'source', BASE / 'runtime', BASE / 'bridge'
    shutil.copytree(SOURCE / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    path = source / 'src/Lokad.Onnx/TensorOps.ConvPool.cs'
    before = path.read_text(encoding='utf8')
    old = '''                var wView = new DenseTensor<float>(wMem.Slice(g * tileM * tileKg, tileM * tileKg), new int[] { tileM, tileKg });
                var pView = new DenseTensor<float>(patchMem.Slice(g * tileKg * colCount, tileKg * colCount), new int[] { tileKg, colCount });
                var dView = new DenseTensor<float>(outMem.Slice(g * tileM * colCount, tileM * colCount), new int[] { tileM, colCount });
                if (!TryConvPortableRows(wView.Buffer.Span, pView.Buffer.Span, dView.Buffer.Span,
                    tileM, tileKg, colCount, options))
                    Tensor<float>.MatMul2D(wView, pView, dView, options);'''
    new = '''                var weights = wMem.Slice(g * tileM * tileKg, tileM * tileKg);
                var patches = patchMem.Slice(g * tileKg * colCount, tileKg * colCount);
                var destination = outMem.Slice(g * tileM * colCount, tileM * colCount);
                if (!TryConvPortableRows(weights.Span, patches.Span, destination.Span,
                    tileM, tileKg, colCount, options))
                {
                    var wView = new DenseTensor<float>(weights, new int[] { tileM, tileKg });
                    var pView = new DenseTensor<float>(patches, new int[] { tileKg, colCount });
                    var dView = new DenseTensor<float>(destination, new int[] { tileM, colCount });
                    Tensor<float>.MatMul2D(wView, pView, dView, options);
                }'''
    assert before.count(old) == 1
    after = before.replace(old, new)
    path.write_text(after, encoding='utf8')
    (BASE / 'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/src/Lokad.Onnx/TensorOps.ConvPool.cs', tofile='b/src/Lokad.Onnx/TensorOps.ConvPool.cs')), encoding='utf8')
    changed = []
    for p in source.rglob('*'):
        if p.is_file():
            original = SOURCE / 'source' / p.relative_to(source)
            if pin(p) != pin(original):
                changed.append(p.relative_to(source).as_posix())
            files[rel(p)] = pin(p)
            files[rel(original)] = pin(original)
    assert changed == ['src/Lokad.Onnx/TensorOps.ConvPool.cs']
    shutil.copytree(BUILD / 'bridge', bridge, ignore=shutil.ignore_patterns('bin', 'obj'))
    path = bridge / 'Program.cs'
    text = path.read_text(encoding='utf8')
    start, end = text.index('    const string panel ='), text.index('    if (!allowed)')
    text = text[:start] + '''    const string tensor = "Lokad.Onnx.Tensor`1[T]::";
    bool allowed = name == "Lokad.Onnx.dll"
        ? removed.Length == 0 && added.Length == 0 && differences.Length == 1
          && differences[0].StartsWith(tensor + "RunTiledBatchFloat::")
        : removed.Length == 0 && added.Length == 0 && differences.Length == 0;
''' + text[end:]
    text = text.replace('equal_except_storage_admission', 'equal_except_deferred_views')
    path.write_text(text, encoding='utf8')
    for p in [*TOOLS.iterdir(), *bridge.iterdir(), MONITOR, BASE / 'candidate.patch',
        ROOT / 'tests/pyannote/portable-integration-tests/common.py']:
        if p.is_file():
            files[rel(p)] = pin(p)
    save(BASE / 'source-prepared.json', dict(passed=True, files=files, changed=changed, normal_project_references=True))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    suites = []

    def run(name, args, test, output):
        monitor.worker(state, BASE / 'processes.json', name, args, source, [0], 10 if test else 8, 8, 900, True, output)
        print(name, 'passed', flush=True)

    try:
        projects = {name: source / path for name, path in [('cli', 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
            ('backend', 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
            ('tensors', 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]}
        projects['bridge'] = bridge / 'Bridge.csproj'
        for name, project in projects.items():
            run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], False, None)
            run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, None)
        shutil.copytree(projects['cli'].parent / 'bin/Release/net10.0', runtime)
        for name in ['cli', 'backend', 'tensors']:
            for assembly in ['Lokad.Onnx.dll'] + ([] if name == 'tensors' else ['Lokad.Onnx.Data.dll']):
                assert pin(projects[name].parent / 'bin/Release/net10.0' / assembly) == pin(runtime / assembly)
        run('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', PRIOR / 'runtime', runtime, BASE / 'instructions.json'], False, bridge)
        assert read(BASE / 'instructions.json')['passed']
        for name, pattern, disabled, passed, skipped in SUITES:
            original = monitor.clean_env
            if disabled:
                def environment():
                    env = original()
                    env['DOTNET_EnableHWIntrinsic'] = '0'
                    return env
                monitor.clean_env = environment
            try:
                project = projects['tensors' if name == 'tensors-full' else 'backend']
                args = ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                    '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results']
                if pattern:
                    args += ['--filter', pattern]
                run(name, args, True, BASE / 'test-results')
            finally:
                monitor.clean_env = original
            suites.append(read_suite(name, passed, skipped))
            save(BASE / 'suites.json', suites)
        verify(files)
        for folder in [source, runtime, bridge]:
            for p in folder.rglob('*'):
                if p.is_file() and 'obj' not in p.relative_to(folder).parts:
                    files[rel(p)] = pin(p)
        save(BASE / 'prepared.json', dict(passed=True, files=files, suites=suites, changed=changed,
            core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
            scope='One-method wrapper deferral; normal source build and complete suites. Graph/public/comparison qualification pending.'))
        state['code'] = 0
        print(dict(prepared=pin(BASE / 'prepared.json'), core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
