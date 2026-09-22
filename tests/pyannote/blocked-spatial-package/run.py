"""Consume the actual normal-source package through public graph execution."""
import importlib.util
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-package-20260922'
PRODUCT = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
PROBE = ROOT/'tests/pyannote/portable-integration/PackageProbeV2.cs'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('blocked_package_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal


def prior():
    proof = read(PRODUCT/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(PRODUCT/name) == wanted, name
    for identity in proof['identities']: terminal(identity)
    result = read(PRODUCT/'analysis.json'); assert result['passed'] and result['product_source_identical']
    assert [(r['passed'], r['skipped']) for r in result['suites']] == [(3344, 93), (343, 0), (31, 0)]
    return result


def package():
    path = BASE/'nuget/Lokad.Onnx.0.2.0.nupkg'
    with zipfile.ZipFile(path) as archive:
        assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (PRODUCT/'runtime/Lokad.Onnx.dll').read_bytes()
        document = ET.fromstring(archive.read('Lokad.Onnx.nuspec'))
        dependencies = [n.attrib for n in document.iter() if n.tag.split('}')[-1] == 'dependency']
        assert len(dependencies) == 1 and dependencies[0]['id'] == 'Google.Protobuf' and dependencies[0]['version'] == '3.33.5'
        return dict(passed=True, package=pin(path), entries=archive.namelist(), dependencies=dependencies,
            core=pin(PRODUCT/'runtime/Lokad.Onnx.dll'))


def main():
    assert not BASE.exists(); product = prior()
    BASE.mkdir(); (BASE/'logs').mkdir(); consumer = BASE/'consumer'; consumer.mkdir()
    project = consumer/'PackageProbe.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><PackageReference Include="Lokad.Onnx" Version="0.2.0" /></ItemGroup></Project>''', encoding='utf8')
    program = PROBE.read_text(); assert program.count('var result = new {') == 1
    program = program.replace('var result = new {', (TOOLS/'Extra.cs.txt').read_text()+
        '\nvar result = new { prepared_graph_values = expectedBlocked.Length, prepared_graph_calls = 2, retained_weights = blockedGraph.RetainedPackedWeightBytes, graph_scratch = blockedContext.LastScratchBytes,')
    (consumer/'Program.cs').write_text(program, encoding='utf8')
    files = {p.as_posix(): pin(p) for p in [*TOOLS.glob('*'), *consumer.glob('*'), PROBE, MONITOR, PRODUCT/'closed.json'] if p.is_file()}
    save(BASE/'inputs.json', dict(files=files, core=product['core'], package_only=True))
    own = monitor.psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE/'controller.json'; flags = monitor.FLAGS+['-p:NuGetAudit=false']

    def run(name, args, numerical=False):
        monitor.worker(state, path, name, args, ROOT, [0], 12 if numerical else 8, 8, 900, True, BASE/'consumer')
        print(name, 'passed', flush=True)

    try:
        run('package', ['dotnet', 'pack', PRODUCT/'source/src/Lokad.Onnx/Lokad.Onnx.csproj', '-c', 'Release', *flags,
            '--no-build', '--no-restore', '--output', BASE/'nuget'])
        save(BASE/'package.json', package())
        run('restore', ['dotnet', 'restore', project, *flags, '--source', BASE/'nuget', '--source', FEED, '--packages', BASE/'packages'])
        run('build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'])
        built = consumer/'bin/Release/net10.0'; assert pin(built/'Lokad.Onnx.dll') == product['core']
        run('consumer', ['dotnet', built/'PackageProbe.dll', PRODUCT/'source/tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx',
            product['core']['sha256'], BASE/'consumer.json'], True)
        result = read(BASE/'consumer.json')
        assert result['passed'] and result['prepared_graph_values'] == 1056 and result['prepared_graph_calls'] == 2
        assert result['retained_weights'] == 18432 and result['graph_scratch'] == 8384
        verify(files); prior()
        save(BASE/'verified.json', dict(passed=True, files=files, core=product['core'], package=package(),
            result=pin(BASE/'consumer.json'), consumer=pin(built/'PackageProbe.dll'), no_performance_measurement=True))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
