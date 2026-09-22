"""Prove single-panel layout and qualify the distinct component before timing."""
import importlib.util
import json
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-single-panel-direct-20260922'
PRIOR = ROOT / 'artifacts/pyannote-two-column-20260922'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('single_panel_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil
CORE = 'e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838'
OLD = '        MathOps.PackPanelsB(n, k, b, packed);'
NEW = '''        // A single full panel or a pure tail already has the row-major layout.
        if (candidate && portable && k <= 32) packed = b;
        else MathOps.PackPanelsB(n, k, b, packed);'''


def rel(path): return path.relative_to(ROOT).as_posix()


def consumer(source):
    assert source.count(OLD) == 1
    return source.replace(OLD, NEW)


def main():
    assert not BASE.exists()
    assert pin(PRIOR / 'closed.json')['sha256'] == '848b9681e1914818a5a9a0a63e98f22b7d1abf06dbc906d243d0527094df2e39'
    closed = read(PRIOR / 'closed.json'); assert closed['passed']; verify(closed['files'])
    assert pin(PRIOR / 'payload/runtime/Lokad.Onnx.dll')['sha256'] == CORE
    BASE.mkdir(); (BASE / 'logs').mkdir(); (BASE / 'output').mkdir(); (BASE / 'runtime').mkdir()
    shutil.copy2(PRIOR / 'payload/shapes.json', BASE / 'shapes.json')
    shutil.copy2(PRIOR / 'payload/runtime/Lokad.Onnx.dll', BASE / 'runtime/Lokad.Onnx.dll')
    projects = {}
    for name, folder, assembly, sources in [
        ('normal', 'normal', 'DirectOutputProbe', ['Probe.cs', 'DirectOutput.cs']),
        ('scalar', 'consumer', 'ScalarTailProbe', ['Probe.cs', 'ScalarDirectOutput.cs', 'ScalarOriginal.cs']),
        ('layout', None, 'PackingLayout', ['Layout.cs'])]:
        target = BASE / name; target.mkdir()
        for file in sources:
            if folder is None: shutil.copy2(TOOLS / file, target / file)
            elif file == 'Probe.cs':
                (target / file).write_text(consumer((PRIOR / folder / file).read_text(encoding='utf8')), encoding='utf8')
            else: shutil.copy2(PRIOR / folder / file, target / file)
        project = target / (assembly + '.csproj')
        includes = ''.join('<Compile Include="' + file + '"/>' for file in sources)
        project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><EnableDefaultCompileItems>false</EnableDefaultCompileItems><AssemblyName>{assembly}</AssemblyName><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup>{includes}<Reference Include="Lokad.Onnx"><HintPath>{BASE / 'runtime/Lokad.Onnx.dll'}</HintPath></Reference></ItemGroup></Project>''', encoding='utf8')
        projects[name] = (project, assembly)
    inputs = {rel(p): pin(p) for folder in [TOOLS, BASE] for p in folder.rglob('*') if p.is_file()}
    inputs.update({rel(p): pin(p) for p in [MONITOR, PRIOR / 'closed.json']})
    save(BASE / 'inputs.json', dict(files=inputs))
    own = psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE / 'preparation.json'; save(path, state)
    def run(name, command, inference):
        monitor.worker(state, path, name, command, ROOT, [0], 8, 4, 900, not inference, BASE / 'output')
        print(name, 'passed', flush=True)
    try:
        flags = monitor.FLAGS + ['-p:NuGetAudit=false']
        for name, (project, assembly) in projects.items():
            run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], False)
            run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False)
            binary = project.parent / 'bin/Release/net10.0'
            assert pin(binary / 'Lokad.Onnx.dll')['sha256'] == CORE
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
                shutil.copy2(binary / (assembly + '.' + suffix), BASE / 'runtime' / (assembly + '.' + suffix))
        for mode in ['normal', 'disabled']:
            clean = monitor.clean_env
            if mode == 'disabled':
                def environment():
                    value = clean(); value['DOTNET_EnableHWIntrinsic'] = '0'; return value
                monitor.clean_env = environment
            try:
                run('layout-' + mode, ['dotnet', BASE / 'runtime/PackingLayout.dll', mode, BASE / 'output' / ('layout-' + mode + '.json')], True)
            finally: monitor.clean_env = clean
        for name, assembly in [('normal', 'DirectOutputProbe'), ('scalar', 'ScalarTailProbe')]:
            run('validate-' + name, ['dotnet', BASE / 'runtime' / (assembly + '.dll'), BASE / 'shapes.json', 'validate',
                                    BASE / 'output' / ('validate-' + name + '.json')], True)
            result = read(BASE / 'output' / ('validate-' + name + '.json'))
            assert result['passed'] and len(result['records']) == 3266
        verify(inputs)
        save(BASE / 'prepared.json', dict(passed=True, inputs=pin(BASE / 'inputs.json'),
             files={rel(p): pin(p) for folder in [BASE / 'runtime', BASE / 'output'] for p in folder.rglob('*') if p.is_file()},
             inference_scope='Layout proof and component correctness only; no new timing or normal product composition'))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
