"""Remove the old Data DLL fixture, compiling only the changed tests."""
import difflib
import shutil
import traceback
from common import *


def main():
    assert not BASE.exists()
    path = COMPLETE / 'closed.json'
    assert pin(path)['sha256'] == 'c07358dee34e3983ab7c212488772deaf0c973cc45274c5b505ac766e27753b4'
    proof = read(path)
    assert proof['passed']
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source = BASE / 'source'
    shutil.copytree(PRIOR / 'source', source, ignore=shutil.ignore_patterns('dense-reference'))
    reference = PRIOR / 'evidence-inputs/src/Lokad.Onnx.Data/WeSpeakerAudio.cs'
    dense = reference.read_text(encoding='utf-8-sig')
    expected = dense.replace('namespace Lokad.Onnx;', 'namespace Lokad.Onnx.Backend.Tests;').replace(
        'public static class WeSpeakerAudio', 'internal static class DenseWeSpeakerReference')
    assert expected == (TOOLS / 'DenseWeSpeakerReference.cs').read_text(encoding='utf8')
    assert 'MelSupport' not in dense and 'k < FourierSize / 2; k++' in dense
    changes = []
    for name in ['SparseMelTests.cs', 'DenseWeSpeakerReference.cs']:
        target = source / 'tests/Lokad.Onnx.Backend.Tests' / name
        before = target.read_text(encoding='utf8') if target.exists() else ''
        after = (TOOLS / name).read_text(encoding='utf8')
        changes.append(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='a/tests/Lokad.Onnx.Backend.Tests/' + name if target.exists() else '/dev/null',
            tofile='b/tests/Lokad.Onnx.Backend.Tests/' + name)))
        shutil.copy2(TOOLS / name, target)
    (BASE / 'tests.patch').write_text(''.join(changes), encoding='utf8')
    backend = source / 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
    assert '<HintPath>' not in backend.read_text() and '<ProjectReference' in backend.read_text()
    assert not list(source.rglob('dense-reference'))
    products = {}
    for p in (source / 'src').rglob('*'):
        if p.is_file() and 'obj' not in p.relative_to(source).parts:
            original = PRIOR / 'source' / p.relative_to(source)
            assert pin(p) == pin(original)
            products[rel(p)] = pin(p)
    files = {rel(path): pin(path), rel(reference): pin(reference), rel(MONITOR): pin(MONITOR),
        rel(BASE / 'tests.patch'): pin(BASE / 'tests.patch'), **products}
    for p in [*TOOLS.iterdir(), *source.rglob('*')]:
        if p.is_file() and (not p.is_relative_to(source) or not {'obj', 'bin'}.intersection(p.relative_to(source).parts)):
            files[rel(p)] = pin(p)
    save(BASE / 'source-prepared.json', dict(passed=True, files=files, products=products,
        reference=pin(reference), core=pin(PRIOR / 'runtime/Lokad.Onnx.dll'), data=pin(PRIOR / 'runtime/Lokad.Onnx.Data.dll'),
        scope='Only test source changes; original project references retained, all product bytes frozen, old dense Data DLL deliberately absent.'))
    owner = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=owner.pid, birth=owner.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    suites = []

    def run(name, args, test):
        monitor.worker(state, BASE / 'processes.json', name, args, source, [0], 10 if test else 8, 8, 900, True,
            BASE / 'test-results' if test else backend.parent / 'bin')
        verify(products)
        print(name, 'passed', flush=True)

    try:
        run('backend-build', ['dotnet', 'build', backend, '-c', 'Release', *flags, '--no-restore',
            '--disable-build-servers', '-p:BuildProjectReferences=false'], False)
        for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']:
            assert pin(backend.parent / 'bin/Release/net10.0' / name) == pin(PRIOR / 'runtime' / name)
        assert not list(source.rglob('dense-reference'))
        for name, pattern, disabled, passed, skipped in [
            ('focused', 'FullyQualifiedName~WeSpeakerAudioTests|FullyQualifiedName~SparseMelTests', False, 89, 0),
            ('hardware-disabled', 'FullyQualifiedName~WeSpeakerAudioTests|FullyQualifiedName~SparseMelTests', True, 89, 0),
            ('backend-full', None, False, 3290, 93), ('tensors-full', None, False, 342, 0)]:
            project = source / 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj' if name == 'tensors-full' else backend
            original = monitor.clean_env
            if disabled:
                def environment():
                    env = original()
                    env['DOTNET_EnableHWIntrinsic'] = '0'
                    return env
                monitor.clean_env = environment
            try:
                args = ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                    '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results']
                if pattern:
                    args += ['--filter', pattern]
                run(name, args, True)
            finally:
                monitor.clean_env = original
            suites.append(read_suite(name, passed, skipped))
            save(BASE / 'suites.json', suites)
        verify(files)
        assert not list(source.rglob('dense-reference'))
        for p in source.rglob('*'):
            if p.is_file() and 'obj' not in p.relative_to(source).parts:
                files[rel(p)] = pin(p)
        state['code'] = 0
        save(BASE / 'prepared.json', dict(passed=True, files=files, products=products, suites=suites,
            core=pin(PRIOR / 'runtime/Lokad.Onnx.dll'), data=pin(PRIOR / 'runtime/Lokad.Onnx.Data.dll'),
            scope='Self-contained test source qualified against byte-identical integrated product; no new product build, package or timing.'))
        print(dict(prepared=pin(BASE / 'prepared.json')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
