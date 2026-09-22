"""Integrate the four reviewed files only after the complete AMD verdict passes."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-single-panel-root-20260922'
AMD = ROOT / 'artifacts/pyannote-single-panel-amd-execution-v2-20260922'
PAYLOAD = ROOT / 'artifacts/pyannote-single-panel-amd-payload-v2-20260922/payload'
MODEL = ROOT / 'artifacts/pyannote-single-panel-composition-20260922'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
NAMES = ['src/Lokad.Onnx/TensorOps.ConvPool.cs', 'src/Lokad.Onnx/Zzz.ConvDirectOutput.cs',
         'src/Lokad.Onnx/Zzz.ConvDirectOutputKernel.cs', 'tests/Lokad.Onnx.Backend.Tests/ConvDirectOutputTests.cs']
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('direct_root_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path): return path.relative_to(ROOT).as_posix()


def admission():
    closed = read(AMD / 'closed.json'); assert closed['passed']
    for name, wanted in closed['files'].items(): assert pin(AMD / name) == wanted, name
    state = read(AMD / 'controller/state.json'); assert state['complete'] and state['code'] == 0
    for identity in [state['supervisor']] + [r['child'] for r in state['stages']]: terminal(identity)
    collection = read(AMD / 'collected/collection.json'); assert collection['terminal'] and collection['input_error'] is None
    analysis = read(AMD / 'analysis.json')
    assert analysis['passed'] and analysis['meetings']['passed'] and analysis['performance']['admitted']
    sys.path.insert(0, str(ROOT / 'tests/pyannote/single-panel-amd-v2'))
    from admission import evaluate
    from candidate_protocol import gate
    assert evaluate(analysis['table']) == analysis['performance']
    gate(analysis['reports'])
    assert analysis['timing_calls'] == 96 and analysis['measured'] == 72 and analysis['warmup'] == 24
    for role in ['production', 'portable']:
        assert all(v['cases'] == 400 for v in analysis['reports'][role]['callers'].values())
    measured = read(PAYLOAD / 'manifests/portable-pyannote.json')
    for field, name in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll')]:
        assert measured[field] == pin(MODEL / 'runtime' / name)['sha256']
    assert pin(MODEL / 'closed.json')['sha256'] == 'c0de18c73357515e34f1e80b112b87c9efb133e0464cccd2776874cc4610378d'
    proof = read(MODEL / 'closed.json'); assert proof['passed']; verify(proof['files'])
    for identity in proof['identities']: terminal(identity)
    return dict(passed=True, amd_closure=pin(AMD / 'closed.json'), amd_analysis=pin(AMD / 'analysis.json'),
                windows_closure=pin(MODEL / 'closed.json'), patch=pin(MODEL / 'candidate.patch'), names=NAMES)


def suite(name, passed, skipped):
    path = BASE / 'test-results' / (name + '.trx'); document = ET.parse(path)
    rows = document.findall('.//{*}UnitTestResult'); counters = document.find('.//{*}Counters').attrib
    assert len(rows) == int(counters['total']) == passed + skipped
    assert sum(r.attrib['outcome'] == 'Passed' for r in rows) == int(counters['passed']) == passed
    assert sum(r.attrib['outcome'] == 'NotExecuted' for r in rows) == skipped and int(counters['failed']) == 0
    assert all(r.attrib['outcome'] in ['Passed', 'NotExecuted'] for r in rows)
    return dict(name=name, passed=passed, skipped=skipped, trx=pin(path))


def main():
    assert not BASE.exists()
    accepted = admission()
    assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True) == ''
    patch = MODEL / 'candidate.patch'
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=ROOT, check=True)
    assert subprocess.check_output(['git', 'apply', '--numstat', str(patch)], cwd=ROOT, text=True).count('\n') == 4
    BASE.mkdir(); (BASE / 'logs').mkdir()
    before = {}
    for name in NAMES:
        p = ROOT / name; before[name] = pin(p) if p.exists() else None
        if p.exists():
            target = BASE / 'before' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    accepted.update(before=before, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save(BASE / 'admission.json', accepted)
    subprocess.run(['git', 'apply', str(patch)], cwd=ROOT, check=True)
    for name in NAMES:
        assert (ROOT / name).read_text(encoding='utf-8-sig') == (MODEL / 'source' / name).read_text(encoding='utf-8-sig')
    bridge = BASE / 'bridge'; bridge.mkdir()
    shutil.copy2(ROOT / 'tests/pyannote/combined-avx512/Inventory.cs.txt', bridge / 'Program.cs')
    shutil.copy2(MODEL / 'bridge/Bridge.csproj', bridge / 'Bridge.csproj')
    consumer = BASE / 'consumer'; consumer.mkdir()
    shutil.copy2(MODEL / 'package-consumer/Program.cs', consumer / 'Program.cs')
    shutil.copy2(MODEL / 'package-consumer/PackageProbe.csproj', consumer / 'PackageProbe.csproj')
    files = {rel(p): pin(p) for p in [*[ROOT / n for n in NAMES], *TOOLS.glob('*.py'), MONITOR, patch,
              bridge / 'Program.cs', bridge / 'Bridge.csproj', consumer / 'Program.cs', consumer / 'PackageProbe.csproj',
              AMD / 'closed.json', AMD / 'analysis.json', MODEL / 'closed.json', ROOT / 'tests/pyannote/single-panel-amd-v2/admission.py',
              ROOT / 'tests/pyannote/single-panel-amd-v2/candidate_protocol.py']}
    save(BASE / 'inputs.json', dict(passed=True, files=files))
    own = psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    path = BASE / 'processes.json'; save(path, state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']
    def run(name, args, inference, output):
        monitor.worker(state, path, name, args, ROOT, [0], 10 if inference else 8, 8, 900, True, output)
        print(name, 'passed', flush=True)
    try:
        projects = {name: ROOT / p for name, p in [('cli', 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
                    ('backend', 'tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
                    ('tensors', 'tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]}
        projects['bridge'] = bridge / 'Bridge.csproj'
        for name, project in projects.items():
            run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], False, None)
            run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, None)
        runtime = BASE / 'runtime'; shutil.copytree(projects['cli'].parent / 'bin/Release/net10.0', runtime)
        for name in ['cli', 'backend', 'tensors']:
            for dll in ['Lokad.Onnx.dll'] + ([] if name == 'tensors' else ['Lokad.Onnx.Data.dll']):
                assert pin(projects[name].parent / 'bin/Release/net10.0' / dll) == pin(runtime / dll)
        run('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', MODEL / 'runtime', runtime, BASE / 'instructions.json'], False, bridge)
        inventory = read(BASE / 'instructions.json'); assert inventory['inventory_complete']
        assert [(r['assembly'], r['methods']) for r in inventory['observations']] == [('Lokad.Onnx.dll', 3113), ('Lokad.Onnx.Data.dll', 697)]
        for row in inventory['observations']:
            assert row['public_surface_equal'] and not row['added'] and not row['removed'] and not row['differences']
        suites = []
        for name, project, passed, skipped in [('backend', projects['backend'], 3313, 93), ('tensors', projects['tensors'], 343, 0)]:
            run(name + '-tests', ['dotnet', 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                '--logger', 'trx;LogFileName=' + name + '.trx', '--results-directory', BASE / 'test-results'], True, BASE / 'test-results')
            suites.append(suite(name, passed, skipped))
        run('package', ['dotnet', 'pack', ROOT / 'src/Lokad.Onnx/Lokad.Onnx.csproj', '-c', 'Release', *flags,
            '--no-build', '--no-restore', '--output', BASE / 'nuget'], False, BASE / 'nuget')
        package = BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg'
        with zipfile.ZipFile(package) as archive:
            assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (runtime / 'Lokad.Onnx.dll').read_bytes()
            document = ET.fromstring(archive.read('Lokad.Onnx.nuspec'))
            dependencies = [r.attrib for r in document.iter() if r.tag.split('}')[-1] == 'dependency']
            assert len(dependencies) == 1 and dependencies[0]['id'] == 'Google.Protobuf' and dependencies[0]['version'] == '3.33.5'
        project = consumer / 'PackageProbe.csproj'
        run('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', BASE / 'nuget', '--source', FEED,
                                '--packages', BASE / 'consumer-cache'], False, None)
        run('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], False, consumer)
        assert pin(consumer / 'bin/Release/net10.0/Lokad.Onnx.dll') == pin(runtime / 'Lokad.Onnx.dll')
        run('consumer', ['dotnet', consumer / 'bin/Release/net10.0/PackageProbe.dll', ROOT / 'tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx',
                        pin(runtime / 'Lokad.Onnx.dll')['sha256'], BASE / 'consumer.json'], True, consumer)
        value = read(BASE / 'consumer.json')
        assert value['passed'] and value['packaged_tiled_convolution_values'] == 33216
        assert value['packaged_narrow_values'] == 10240 and value['narrow_scratch_bytes'] == 327680
        assert value['model_imported'] and value['input_and_held_outputs_unchanged']
        verify(files)
        save(BASE / 'verified.json', dict(passed=True, files=files, core=pin(runtime / 'Lokad.Onnx.dll'), data=pin(runtime / 'Lokad.Onnx.Data.dll'),
             suites=suites, package=pin(package), dependencies=dependencies, instructions=pin(BASE / 'instructions.json'), no_new_performance_measurement=True))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(path, state)


if __name__ == '__main__': main()
