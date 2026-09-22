"""Qualify actual retained layer calls through the normal graph/provider path."""
import importlib.util
import json
from pathlib import Path
import shutil
import traceback
from transform import transform

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-layer-graphs-20260922'
PRODUCT = ROOT/'artifacts/pyannote-blocked-spatial-composition-20260922'
REVIEW = ROOT/'artifacts/pyannote-blocked-spatial-composition-review-v3-20260922'
COMPONENT = ROOT/'artifacts/pyannote-vector-input-layout-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('graph_layer_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal


def main():
    assert not BASE.exists()
    for folder, filename, sha, passed in [
        (PRODUCT, 'failure-closed.json', 'b52ca1fe7ed0edf06094708ae581c1224cd659a1903ec6844ac58af76856b901', False),
        (REVIEW, 'closed.json', '23816c9ce3b718bca1a89c81e249c382158f05ef434f4dd6d909bcfb060a1d8a', True),
        (COMPONENT, 'closed.json', '8dae29462e8d0694031b41e0db31387da550b0796c2fc386ebecf2f00e21a978', True),
        (FIXTURES, 'closed.json', '27ac5ab76c21828d18e59478f6010a4139df0530b349ba5863e5f771dc97422e', True)]:
        assert pin(folder/filename)['sha256'] == sha
        proof = read(folder/filename); assert proof['passed'] == passed
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        for identity in proof.get('identities', []): terminal(identity)
    product = read(REVIEW/'analysis.json'); assert product['passed'] and product['preparation_only']
    assert pin(PRODUCT/'runtime/Lokad.Onnx.dll') == product['core']
    BASE.mkdir(); (BASE/'logs').mkdir(); (BASE/'output').mkdir(); source = BASE/'source'; source.mkdir()
    shutil.copytree(PRODUCT/'runtime', BASE/'runtime')
    for name in ['BlockedSpatial.cs', 'GeneratedKernels.cs', 'VectorInput.cs', 'VectorEpilogue.cs']:
        shutil.copy2(COMPONENT/'source'/name, source/name)
    text, diff = transform((COMPONENT/'source/ModelProbe.cs').read_text(), product['core']['sha256'])
    (source/'ModelProbe.cs').write_text(text, encoding='utf8'); (BASE/'consumer.diff').write_text(diff, encoding='utf8')
    shutil.copy2(TOOLS/'GraphCalls.cs.txt', source/'GraphCalls.cs')
    project = source/'LayerGraphs.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><StartupObject>ModelProbe</StartupObject><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../runtime/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../runtime/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>''', encoding='utf8')
    files = {p.as_posix(): pin(p) for folder in [source, BASE/'runtime', TOOLS, FIXTURES/'output'] for p in folder.rglob('*') if p.is_file()}
    for p in [PRODUCT/'failure-closed.json', REVIEW/'closed.json', COMPONENT/'closed.json', FIXTURES/'closed.json', BASE/'consumer.diff', MONITOR]: files[p.as_posix()] = pin(p)
    save(BASE/'inputs.json', dict(files=files, core=product['core'], no_performance_measurement=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; save(path, state); flags = monitor.FLAGS+['-p:NuGetAudit=false']
    try:
        monitor.worker(state, path, 'restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE/'packages'], ROOT, [0], 8, 8, 900, True, source)
        monitor.worker(state, path, 'build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], ROOT, [0], 8, 8, 900, True, source)
        built = source/'bin/Release/net10.0'; assert pin(built/'Lokad.Onnx.dll') == product['core']
        monitor.worker(state, path, 'graphs-256', ['dotnet', built/'LayerGraphs.dll', '256', FIXTURES/'output', BASE/'output/256.json'], ROOT, [0, 1], 12, 8, 900, False, BASE/'output')
        result = read(BASE/'output/256.json'); assert result['passed'] == (state['runs'][-1]['code'] == 0)
        state['code'] = 0 if result['passed'] else 1; verify(files)
        save(BASE/'verified.json', dict(passed=result['passed'], files=files, report=pin(BASE/'output/256.json'),
            core=product['core'], consumer=pin(built/'LayerGraphs.dll'), no_performance_measurement=True))
        print(json.dumps(dict(passed=result['passed'], cases=result['cases'], values=result['values'], differences=result['differences'], native_failures=result['native_failures'], graph_dispatch=len(result['graph_dispatch']))))
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
