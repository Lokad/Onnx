"""Build one isolated successor and qualify unchanged raw/model probes locally."""
import importlib.util
import json
from pathlib import Path
import shutil
import traceback
from transform import transform

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-vector-input-layout-20260922'
RAW = ROOT/'artifacts/pyannote-blocked-spatial-raw-v2-20260922'
MODEL = ROOT/'artifacts/pyannote-blocked-spatial-model-20260922'
SCREEN = ROOT/'artifacts/pyannote-blocked-spatial-screen-20260922'
EPILOGUE = ROOT/'artifacts/pyannote-vector-output-epilogue-20260922'
PHASE = ROOT/'artifacts/pyannote-blocked-spatial-phases-amd-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
spec = importlib.util.spec_from_file_location('epilogue_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify = monitor.pin, monitor.read, monitor.save, monitor.verify


def main():
    assert not BASE.exists()
    closures = [(EPILOGUE, '3c09605bcf5a254d4619bc3d8857eda2bb0f6d4c9b5424e1d0db30062834ee69', True), (RAW, '943d8dc7d25fa95c9778e5ce4d62efbf693b9c63c4a2eebf04bb9370593d25b9', False),
        (MODEL, 'bb8f483c6dc6edad3ed6f89f739ccc1fadf7c8cd67ec507e24b617d73c53f54d', True),
        (SCREEN, '1412473e7eae6ae69f42c2f47bdf589b8b765014cc34e96f6167ee1fc96fda32', True),
        (PHASE, '9550dfcd6cc46b1d2556f034f3d3dfe485b7868f1bfe5ff4d5cb2eeaa9f02909', True),
        (FIXTURES, '27ac5ab76c21828d18e59478f6010a4139df0530b349ba5863e5f771dc97422e', True)]
    for folder, sha, relative in closures:
        assert pin(folder/'closed.json')['sha256'] == sha
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin((folder if relative else ROOT)/name) == wanted, name
        for identity in proof.get('identities', []): monitor.terminal(identity)
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'output').mkdir(); (BASE/'tools').mkdir(); source = BASE/'source'; source.mkdir()
    for p in TOOLS.iterdir():
        if p.is_file(): shutil.copy2(p, BASE/'tools'/p.name)
    shutil.copytree(RAW/'runtime', BASE/'runtime')
    for folder, names in [(EPILOGUE, ['BlockedSpatial.cs', 'GeneratedKernels.cs', 'Probe.cs', 'ModelProbe.cs', 'VectorEpilogue.cs'])]:
        for name in names: shutil.copy2(folder/'source'/name, source/name)
    original = (source/'BlockedSpatial.cs').read_text(); candidate, diff = transform(original)
    (source/'BlockedSpatial.cs').write_text(candidate, encoding='utf8'); (BASE/'source.diff').write_text(diff, encoding='utf8')
    for name in ['Entry', 'VectorInput', 'ComponentBench']: shutil.copy2(TOOLS/(name+'.cs.txt'), source/(name+'.cs'))
    project = source/'VectorInputProbe.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><StartupObject>VectorInputProbe</StartupObject><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../runtime/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../runtime/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>''', encoding='utf8')
    shutil.copy2(ROOT/'.agent/m16-pyannote-vector-input-20260922.md', BASE/'prospective-plan.md')
    files = {p.as_posix(): pin(p) for p in [*[p for p in BASE.rglob('*') if p.is_file()], MONITOR, ROOT/'src/Lokad.Onnx/MathOps.cs',
        *[folder/'closed.json' for folder, _, _ in closures], *[p for p in TOOLS.iterdir() if p.is_file()],
        *[p for p in (FIXTURES/'output').glob('*') if p.is_file()]]}
    save(BASE/'inputs.json', dict(files=files, no_performance_measurement=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; save(path, state); flags = monitor.FLAGS+['-p:NuGetAudit=false']
    try:
        monitor.worker(state, path, 'restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'], ROOT, [0], 8, 8, 900, True, source)
        monitor.worker(state, path, 'build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], ROOT, [0], 8, 8, 900, True, source)
        built = source/'bin/Release/net10.0'; assert pin(built/'Lokad.Onnx.dll') == pin(RAW/'runtime/Lokad.Onnx.dll')
        for mode in ['raw', 'model']:
            arguments = [mode, '256'] + ([FIXTURES/'output'] if mode == 'model' else []) + [BASE/'output'/(mode+'-256.json')]
            monitor.worker(state, path, mode+'-256', ['dotnet', built/'VectorInputProbe.dll', *arguments], ROOT, [0, 1], 12, 8, 900, False, BASE/'output')
            result = read(BASE/'output'/(mode+'-256.json')); assert result['passed'] == (state['runs'][-1]['code'] == 0)
            if not result['passed']:
                state['code'] = 1; break
        else: state['code'] = 0
        verify(files)
        reports = {p.name: pin(p) for p in (BASE/'output').glob('*.json')}
        save(BASE/'verified.json', dict(complete=True, passed=state['code'] == 0, files=files, reports=reports,
            core=pin(built/'Lokad.Onnx.dll'), consumer=pin(built/'VectorInputProbe.dll')))
        print(json.dumps(dict(passed=state['code'] == 0, reports=reports)))
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)
    return state['code']


if __name__ == '__main__': raise SystemExit(main())
