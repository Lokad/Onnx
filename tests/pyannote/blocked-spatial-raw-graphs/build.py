"""Build and qualify actual product helpers and the complete raw graph census."""
import importlib.util
import json
from pathlib import Path
import shutil
import traceback
from transform import transform

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-raw-graphs-20260922'
PRODUCT = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
COMPONENT = ROOT/'artifacts/pyannote-vector-input-layout-20260922'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
CORE = '3c2f16b08856426d3dfeff07f1638dd76cee7f06b65bbee230e8e0789679206f'
spec = importlib.util.spec_from_file_location('raw_graph_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal


def prior():
    for folder, sha in [(PRODUCT, 'e7a9a30d88ef2a425c0c51b007e2b7d89428d54e50ef7dc4e446e191f95719e3'),
            (COMPONENT, '8dae29462e8d0694031b41e0db31387da550b0796c2fc386ebecf2f00e21a978')]:
        assert pin(folder/'closed.json')['sha256'] == sha
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        for identity in proof['identities']: terminal(identity)
    assert pin(PRODUCT/'runtime/Lokad.Onnx.dll')['sha256'] == CORE


def main():
    assert not BASE.exists(); prior()
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'output').mkdir(); source = BASE/'source'; source.mkdir()
    shutil.copytree(PRODUCT/'runtime', BASE/'runtime')
    output, diff = transform((COMPONENT/'source/Probe.cs').read_text(), CORE)
    (source/'Probe.cs').write_text(output, encoding='utf8'); (BASE/'consumer.diff').write_text(diff, encoding='utf8')
    shutil.copy2(TOOLS/'GraphRaw.cs.txt', source/'GraphRaw.cs')
    project = source/'RawGraphs.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><StartupObject>Probe</StartupObject><TargetFramework>net10.0</TargetFramework><AssemblyName>Lokad.Onnx.Backend.Tests</AssemblyName><AllowUnsafeBlocks>true</AllowUnsafeBlocks><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../runtime/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../runtime/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>''', encoding='utf8')
    files = {p.as_posix(): pin(p) for folder in [source, TOOLS, BASE/'runtime'] for p in folder.rglob('*') if p.is_file()}
    for p in [MONITOR, BASE/'consumer.diff', PRODUCT/'closed.json', COMPONENT/'closed.json', COMPONENT/'source/Probe.cs']:
        files[p.as_posix()] = pin(p)
    save(BASE/'inputs.json', dict(files=files, core=pin(PRODUCT/'runtime/Lokad.Onnx.dll'), no_performance_measurement=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; flags = monitor.FLAGS+['-p:NuGetAudit=false']
    try:
        monitor.worker(state, path, 'restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'], ROOT, [0], 8, 8, 900, True, source)
        monitor.worker(state, path, 'build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], ROOT, [0], 8, 8, 900, True, source)
        built = source/'bin/Release/net10.0'; assert pin(built/'Lokad.Onnx.dll')['sha256'] == CORE
        monitor.worker(state, path, 'raw-256', ['dotnet', built/'Lokad.Onnx.Backend.Tests.dll', '256', BASE/'output/256.json'], ROOT, [0, 1], 12, 8, 900, False, BASE/'output')
        result = read(BASE/'output/256.json'); assert result['passed'] == (state['runs'][-1]['code'] == 0)
        verify(files); prior()
        save(BASE/'verified.json', dict(passed=result['passed'], files=files, core=pin(built/'Lokad.Onnx.dll'),
            consumer=pin(built/'Lokad.Onnx.Backend.Tests.dll'), report=pin(BASE/'output/256.json'), no_performance_measurement=True))
        state['code'] = 0 if result['passed'] else 1
        print(json.dumps(dict(passed=result['passed'], cases=result['cases'], graph_controls=result['graph_controls'],
            graph_candidates=result['graph_candidates'], differences=result['graph_differences'])))
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
