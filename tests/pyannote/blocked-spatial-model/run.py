"""Build the actual-shape consumer against the unchanged selected runtime."""
import importlib.util
import json
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-model-20260922'
RAW = ROOT/'artifacts/pyannote-blocked-spatial-raw-v2-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
spec = importlib.util.spec_from_file_location('model_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify = monitor.pin, monitor.read, monitor.save, monitor.verify


def main():
    assert not BASE.exists()
    for folder, expected, relative in [(RAW, '943d8dc7d25fa95c9778e5ce4d62efbf693b9c63c4a2eebf04bb9370593d25b9', False),
        (FIXTURES, '27ac5ab76c21828d18e59478f6010a4139df0530b349ba5863e5f771dc97422e', True)]:
        assert pin(folder/'closed.json')['sha256'] == expected
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin((folder if relative else ROOT)/name) == wanted
        for identity in proof['identities']: monitor.terminal(identity)
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'output').mkdir(); (BASE/'tools').mkdir()
    for p in TOOLS.iterdir():
        if p.is_file(): shutil.copy2(p, BASE/'tools'/p.name)
    source = BASE/'source'; source.mkdir(); shutil.copytree(RAW/'runtime', BASE/'runtime')
    for name in ['BlockedSpatial.cs', 'GeneratedKernels.cs']: shutil.copy2(RAW/'source'/name, source/name)
    shutil.copy2(TOOLS/'ModelProbe.cs.txt', source/'ModelProbe.cs')
    project = source/'BlockedSpatialModel.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../runtime/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../runtime/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>''', encoding='utf8')
    shutil.copy2(ROOT/'.agent/m13-pyannote-blocked-spatial-20260922.md', BASE/'prospective-plan.md')
    files = {p.as_posix(): pin(p) for p in [*[p for p in BASE.rglob('*') if p.is_file()], MONITOR, RAW/'closed.json', FIXTURES/'closed.json',
             *[p for p in TOOLS.iterdir() if p.is_file()], *[p for p in (FIXTURES/'output').glob('*') if p.is_file()]]}
    save(BASE/'inputs.json', dict(files=files, no_performance_measurement=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; save(path, state); flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    try:
        monitor.worker(state, path, 'restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'], ROOT, [0], 8, 8, 900, True, source)
        monitor.worker(state, path, 'build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], ROOT, [0], 8, 8, 900, True, source)
        built = source/'bin/Release/net10.0'; assert pin(built/'Lokad.Onnx.dll') == pin(RAW/'runtime/Lokad.Onnx.dll')
        monitor.worker(state, path, 'qualify-256', ['dotnet', built/'BlockedSpatialModel.dll', '256', FIXTURES/'output', BASE/'output/256.json'], ROOT, [0, 1], 12, 8, 900, False, BASE/'output')
        result = read(BASE/'output/256.json'); assert result['cases'] == 108 and result['passed'] == (state['runs'][-1]['code'] == 0)
        verify(files); save(BASE/'verified.json', dict(complete=True, passed=result['passed'], report=pin(BASE/'output/256.json'), files=files,
            core=pin(built/'Lokad.Onnx.dll'), consumer=pin(built/'BlockedSpatialModel.dll')))
        state['code'] = 0 if result['passed'] else 1
        print(json.dumps({k: result[k] for k in ['passed', 'cases', 'differences', 'native_failures', 'maximum', 'failed']}))
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)
    return state['code']


if __name__ == '__main__': raise SystemExit(main())
