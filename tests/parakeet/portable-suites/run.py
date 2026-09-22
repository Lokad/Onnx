"""Normal source builds and package qualification after complete native checks."""
import difflib
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'portable-models'))
from common import ROOT, MODEL, FEED, MONITOR, pin, read, save, verify, terminal, psutil, monitor, rel

BASE = ROOT / 'artifacts/parakeet-portable-suites-20260922'
AUDIO = ROOT / 'artifacts/parakeet-portable-models-20260922'
SHARED = ROOT / 'artifacts/parakeet-portable-shared-20260922'


def suite(name, passed, skipped):
    path = BASE / 'test-results' / (name + '.trx')
    xml = ET.parse(path)
    rows = xml.findall('.//{*}UnitTestResult'); counters = xml.find('.//{*}Counters').attrib
    assert len(rows) == int(counters['total']) == passed + skipped
    assert sum(r.attrib['outcome'] == 'Passed' for r in rows) == int(counters['passed']) == passed
    assert sum(r.attrib['outcome'] == 'NotExecuted' for r in rows) == skipped
    assert all(r.attrib['outcome'] in ['Passed', 'NotExecuted'] for r in rows)
    return dict(name=name, passed=passed, skipped=skipped, trx=pin(path))


def main():
    assert not BASE.exists()
    for folder in [MODEL, AUDIO, SHARED]:
        proof = read(folder / 'closed.json'); assert proof['passed']; verify(proof['files'])
        for identity in proof.get('identities', proof.get('terminal_identities', [])): terminal(identity)
    qualified = read(SHARED / 'analysis.json')
    assert qualified['passed'] and qualified['arrays'] == 166 and qualified['values'] == 5000814
    constant = {r['name']: r for r in qualified['dino_hashes'] if r['scenario'] == 'constant'}
    pair = (constant['last_hidden_state']['hash'], constant['pooler_output']['hash'])
    assert pair == (12398120018957570767, 204586709688817035)
    assert all(r['maximum'] <= 1e-4 for r in qualified['dino_hashes'])
    BASE.mkdir(); (BASE / 'logs').mkdir()
    source = BASE / 'source'; source.mkdir()
    commit = subprocess.check_output(['git', 'rev-parse', '1b36c548'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'archive', '--format=zip', '--output=' + str(BASE / 'source.zip'), commit], cwd=ROOT, check=True)
    with zipfile.ZipFile(BASE / 'source.zip') as archive:
        assert all((source / name).resolve().is_relative_to(source) for name in archive.namelist())
        archive.extractall(source)
    for name in ['TensorOps.MatMul.cs', 'MathOps.PartialReduction.cs']:
        shutil.copy2(MODEL / 'source/src/Lokad.Onnx' / name, source / 'src/Lokad.Onnx' / name)
    dino = source / 'tests/Lokad.Onnx.Backend.Tests/GraphExecutionDinoV3Tests.cs'
    before = dino.read_text(encoding='utf8')
    marker = '            (12756423648221837382UL, 6147948217399512682UL)'
    assert before.count(marker) == 1
    after = before.replace(marker, marker + ',\n            // Every output independently checked against ORT at the unchanged 1e-4 bound.\n            (12398120018957570767UL, 204586709688817035UL)')
    dino.write_text(after, encoding='utf8')
    save(BASE / 'test-change.json', dict(path=dino.relative_to(source).as_posix(), shared_closure=pin(SHARED / 'closed.json'),
        reason='Add independently native-qualified complete hash pair; preserve all prior pairs and numerical assertions',
        diff=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))))
    files = {rel(p): pin(p) for p in source.rglob('*') if p.is_file()}
    bridge = MODEL / 'bridge/bin/Release/net10.0'
    for p in [Path(__file__).resolve(), Path(__file__).with_name('audit.py'), MONITOR,
              ROOT / 'tests/parakeet/portable-models/common.py', ROOT / 'tests/pyannote/portable-integration/PackageProbe.cs',
              MODEL / 'closed.json', AUDIO / 'closed.json', SHARED / 'closed.json',
              *[p for p in bridge.iterdir() if p.is_file()]]:
        files[rel(p)] = pin(p)
    save(BASE / 'inputs.json', dict(passed=True, files=files, source_commit=commit))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    monitor.BASE = BASE
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    def run(name, command, inference, output):
        monitor.worker(state, BASE / 'processes.json', name, command, source, [0], 10 if inference else 8, 8, 900, True, output)
        print(name, 'passed', flush=True)
    try:
        projects = {name: source / path for name, path in [('cli', 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
            ('backend', 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
            ('tensors', 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]}
        for name, project in projects.items():
            assert '<ProjectReference' in project.read_text() and '<HintPath>' not in project.read_text()
            run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], False, None)
            run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, project.parent / 'bin')
        runtime = BASE / 'runtime'; shutil.copytree(projects['cli'].parent / 'bin/Release/net10.0', runtime)
        for name in ['cli', 'backend', 'tensors']:
            for assembly in ['Lokad.Onnx.dll'] + ([] if name == 'tensors' else ['Lokad.Onnx.Data.dll']):
                assert pin(projects[name].parent / 'bin/Release/net10.0' / assembly) == pin(runtime / assembly)
        run('instructions', ['dotnet', bridge / 'Bridge.dll', MODEL / 'runtime', runtime, BASE / 'instructions.json'], False, None)
        inventory = read(BASE / 'instructions.json'); assert inventory['inventory_complete']
        assert [(r['assembly'], r['methods']) for r in inventory['observations']] == [('Lokad.Onnx.dll', 3109), ('Lokad.Onnx.Data.dll', 697)]
        for row in inventory['observations']:
            assert row['public_surface_equal'] and not row['added'] and not row['removed'] and not row['differences']
        suites = []
        for name, passed, skipped in [('backend', 3290, 93), ('tensors', 343, 0)]:
            run(name + '-tests', ['dotnet', 'test', projects[name], '-c', 'Release', *flags, '--no-build', '--no-restore',
                '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results'], True, BASE / 'test-results')
            suites.append(suite(name, passed, skipped))
        run('package', ['dotnet', 'pack', source / 'src/Lokad.Onnx/Lokad.Onnx.csproj', '-c', 'Release', *flags,
            '--no-build', '--no-restore', '--output', BASE / 'nuget'], False, BASE / 'nuget')
        package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
        with zipfile.ZipFile(package) as archive:
            assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (runtime / 'Lokad.Onnx.dll').read_bytes()
        consumer = BASE / 'consumer'; consumer.mkdir()
        project = consumer / 'PackageProbe.csproj'
        project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework></PropertyGroup><ItemGroup><PackageReference Include="Lokad.Onnx" Version="0.2.0" /></ItemGroup></Project>\n', encoding='utf8')
        shutil.copy2(ROOT / 'tests/pyannote/portable-integration/PackageProbe.cs', consumer / 'Program.cs')
        run('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', BASE / 'nuget', '--source', FEED,
            '--packages', BASE / 'consumer-cache'], False, None)
        run('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, consumer)
        assert pin(consumer / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(runtime / 'Lokad.Onnx.dll')
        run('consumer', ['dotnet', consumer / 'bin/Release/net10.0/PackageProbe.dll',
            ROOT / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx', pin(runtime / 'Lokad.Onnx.dll')['sha256'], BASE / 'consumer.json'], True, consumer)
        assert read(BASE / 'consumer.json')['passed']
        verify(files)
        save(BASE / 'verified.json', dict(passed=True, files=files, suites=suites, package=pin(package),
            core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
            instruction_equivalent=True, public_surface_equal=True, production_changed=False, timing_selected=False))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'processes.json', state)


if __name__ == '__main__': main()
