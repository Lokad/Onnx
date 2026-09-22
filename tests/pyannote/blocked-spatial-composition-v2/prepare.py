"""Build a fresh normal source tree with corrected test helpers; qualify all suites."""
import shutil
import traceback
from common import *
from transform import transform

SUITES = [('backend-full', 'Backend', 3344, 93, None, False), ('tensors-full', 'Tensors', 343, 0, None, False),
          ('focused-disabled', 'Backend', 31, 0, 'FullyQualifiedName~ConvBlockedSpatialTests', True)]


def main():
    assert not BASE.exists(); close_failure(); priors()
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'test-results').mkdir()
    source = BASE/'source'
    shutil.copytree(PRODUCT/'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    relative = Path('tests/Lokad.Onnx.Backend.Tests/ConvBlockedSpatialTests.cs')
    corrected = (FOCUSED/'focused/ConvBlockedSpatialTests.cs').read_text()
    output, diff = transform(corrected)
    (source/relative).write_text(output, encoding='utf8'); (BASE/'test-helper.diff').write_text(diff, encoding='utf8')
    for p in source.rglob('*'):
        if p.is_file() and p.relative_to(source) != relative: assert pin(p) == pin(PRODUCT/'source'/p.relative_to(source))
    files = {p.as_posix(): pin(p) for folder in [source, TOOLS] for p in folder.rglob('*') if p.is_file()}
    for p in [MONITOR, BASE/'test-helper.diff', FOCUSED/'closed.json', FOCUSED/'focused/ConvBlockedSpatialTests.cs',
              PRODUCT/'failure-closed.json', FAILED/'failure-closed.json', PRODUCT/'bridge/bin/Release/net10.0/Bridge.dll']:
        files[p.as_posix()] = pin(p)
    save(BASE/'inputs.json', dict(files=files, product_source_identical=True, only_changed_source=relative.as_posix(), suites=SUITES))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; flags = monitor.FLAGS+['-p:NuGetAudit=false']

    def run(name, args, numerical=False):
        monitor.worker(state, path, name, args, source, [0], 12 if numerical else 8, 8, 900, True, None)
        print(name, 'passed', flush=True)

    try:
        for name, project in [('cli', source/'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
                ('backend', source/'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
                ('tensors', source/'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]:
            run(name+'-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'])
            run(name+'-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        shutil.copytree(source/'src/Lokad.Onnx.CLI/bin/Release/net10.0', BASE/'runtime')
        for kind in ['Backend', 'Tensors']:
            assert pin(source/f'tests/Lokad.Onnx.{kind}.Tests/bin/Release/net10.0/Lokad.Onnx.dll') == pin(BASE/'runtime/Lokad.Onnx.dll')
        run('inventory', ['dotnet', PRODUCT/'bridge/bin/Release/net10.0/Bridge.dll', PRODUCT/'runtime', BASE/'runtime', BASE/'instructions.json',
            ROOT/'artifacts/pyannote-vector-input-layout-20260922/source/bin/Release/net10.0'])
        save(BASE/'instruction-review.json', inventory())
        outcomes = []
        for name, kind, passed, skipped, pattern, disabled in SUITES:
            project = source/f'tests/Lokad.Onnx.{kind}.Tests/Lokad.Onnx.{kind}.Tests.csproj'
            args = ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                '--logger', 'trx;LogFileName='+name+'.trx', '--results-directory', BASE/'test-results']
            if pattern: args += ['--filter', pattern]
            clean = monitor.clean_env
            if disabled: monitor.clean_env = lambda: clean() | {'DOTNET_EnableHWIntrinsic': '0'}
            try: run(name, args, True)
            finally: monitor.clean_env = clean
            outcomes.append(suite(BASE, name, passed, skipped, 0)); save(BASE/'suites.json', outcomes)
        verify(files); priors()
        save(BASE/'verified.json', dict(passed=True, files=files, suites=outcomes, instruction_review=inventory(),
            core=pin(BASE/'runtime/Lokad.Onnx.dll'), data=pin(BASE/'runtime/Lokad.Onnx.Data.dll'),
            product_source_identical=True, package_qualified=False, models_qualified=False, performance_qualified=False))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
