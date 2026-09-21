"""Freeze the tracer and bounded no-model attach/export probe."""
from common import *


def main():
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    state = new_state()
    save(BASE / 'preparation.json', state)
    try:
        tracer = BASE / 'tracer'
        shutil.copytree(TRACE_SOURCE, tracer)
        external = {str(p): pin(p) for p in TRACE_SOURCE.rglob('*') if p.is_file()}
        shim = Path('C:/Users/JoannesVermorel/.dotnet/tools/dotnet-trace.exe')
        external[str(shim)] = pin(shim)
        source = BASE / 'toy-source'
        source.mkdir()
        shutil.copy2(TOOLS / 'Toy.cs', source / 'Program.cs')
        (source / 'Toy.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup></Project>\n', encoding='utf8')
        flags = monitor.FLAGS + ['-p:NuGetAudit=false']
        for name, command in [
            ('version', ['dotnet', tracer / 'dotnet-trace.dll', '--version']),
            ('profiles', ['dotnet', tracer / 'dotnet-trace.dll', 'list-profiles']),
            ('collect-help', ['dotnet', tracer / 'dotnet-trace.dll', 'collect', '--help']),
            ('convert-help', ['dotnet', tracer / 'dotnet-trace.dll', 'convert', '--help']),
            ('restore', ['dotnet', 'restore', source / 'Toy.csproj', *flags, '--source', FEED, '--packages', BASE / 'packages']),
            ('build', ['dotnet', 'build', source / 'Toy.csproj', '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])]:
            monitor.worker(state, BASE / 'preparation.json', name, command, ROOT, [0], 8, 8, 900, True, None)
        assert (BASE / 'logs/version.log').read_text().strip() == '10.0.745401+cef304c50763bf24f99566cb31d55540842e7ae9'
        files = {rel(p): pin(p) for folder in [tracer, source, TOOLS] for p in folder.rglob('*') if p.is_file() and 'obj' not in p.relative_to(folder).parts}
        files[rel(MONITOR)] = pin(MONITOR)
        save(BASE / 'prepared.json', dict(passed=True, files=files, external_files=external,
            tool_version='10.0.745401+cef304c50763bf24f99566cb31d55540842e7ae9',
            limits=dict(preflight_gib=8, rss_gib=8, available_gib=1, disk_gib=20, output_gib=1, seconds=900),
            source='https://learn.microsoft.com/en-us/dotnet/core/diagnostics/dotnet-trace',
            scope='Toy attach/export qualification only; managed sampled thread time, not kernel CPU sampling.'))
        state['code'] = 0
        print(dict(prepared=pin(BASE / 'prepared.json')), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'preparation.json', state)


if __name__ == '__main__':
    main()
