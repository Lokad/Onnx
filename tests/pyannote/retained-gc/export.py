"""Build an offline reader and export the two existing complete-request traces."""
import shutil
import traceback
from common import *


def main():
    assert not BASE.exists()
    closed = INPUT / 'model-closed.json'
    assert pin(closed)['sha256'] == '71c2e9c36f31216fe538454426aea9a15d7beaa7a19f72c038aaa7fec6bf818f'
    proof = read(closed)
    assert proof['passed']
    verify(proof['files'])
    for name, expected in proof.get('external_files', {}).items():
        assert pin(Path(name)) == expected
    for identity in proof['identities']:
        terminal(identity)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    source = BASE / 'source'
    source.mkdir()
    shutil.copy2(TOOLS / 'Export.cs', source / 'Program.cs')
    references = ['Microsoft.Diagnostics.Tracing.TraceEvent', 'Microsoft.Diagnostics.FastSerialization', 'Microsoft.Diagnostics.NETCore.Client']
    refs = '\n'.join(f'<Reference Include="{name}"><HintPath>{INPUT / "tracer" / (name + ".dll")}</HintPath></Reference>' for name in references)
    project = source / 'Export.csproj'
    project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework>'
        '<ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup>' + refs + '</ItemGroup></Project>', encoding='utf8')
    files = {rel(closed): pin(closed), rel(MONITOR): pin(MONITOR)}
    for path in [*TOOLS.iterdir(), *source.iterdir(), *[INPUT / 'tracer' / (name + '.dll') for name in references]]:
        if path.is_file():
            files[rel(path)] = pin(path)
    for name in ['sampled-a', 'sampled-b']:
        for stem in ['capture.nettrace', 'result.json', 'ready.json']:
            path = INPUT / name / stem
            files[rel(path)] = pin(path)
    save(BASE / 'inputs.json', dict(passed=True, files=files, inference_executed=False))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    def run(name, args, preflight, children, output):
        monitor.worker(state, BASE / 'processes.json', name, args, source, [0], preflight, 2, 300, children, output)
        print(name, 'passed', flush=True)
    try:
        run('restore', ['dotnet', 'restore', project, *flags, '--source', source], 8, True, source)
        run('build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], 8, True, source)
        runtime = source / 'bin/Release/net10.0'
        for name in references:
            assert pin(runtime / (name + '.dll')) == pin(INPUT / 'tracer' / (name + '.dll'))
        for path in runtime.iterdir():
            if path.is_file():
                files[rel(path)] = pin(path)
        save(BASE / 'prepared.json', dict(passed=True, files=files, inference_executed=False))
        for name in ['sampled-a', 'sampled-b']:
            run(name, ['dotnet', runtime / 'Export.dll', INPUT / name / 'capture.nettrace', BASE / name], 2, False, BASE / name)
        verify(files)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
