"""Integrate the three qualified arithmetic/test files only after the complete AMD verdict passes."""
import difflib
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
BASE = ROOT / 'artifacts/parakeet-single-panel-root-20260922'
AMD = ROOT / 'artifacts/parakeet-single-panel-amd-execution-v2-20260922'
PAYLOAD = ROOT / 'artifacts/parakeet-single-panel-amd-payload-v2-20260922/payload'
MODEL = ROOT / 'artifacts/parakeet-single-panel-composition-20260922'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
QUALIFIED = ROOT / 'artifacts/parakeet-single-panel-suites-20260922'
SELECTED = ROOT / 'artifacts/pyannote-single-panel-root-20260922'
NAMES = ['src/Lokad.Onnx/TensorOps.MatMul.cs', 'src/Lokad.Onnx/MathOps.PartialReduction.cs',
         'tests/Lokad.Onnx.Backend.Tests/GraphExecutionDinoV3Tests.cs']
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
    sys.path.insert(0, str(ROOT / 'tests/parakeet/single-panel-amd-v2'))
    from admission import evaluate
    from candidate_protocol import gate
    assert evaluate(analysis['table']) == analysis['performance']
    gate(analysis['reports'])
    assert analysis['timing_calls'] == 480 and analysis['measured'] == 360 and analysis['warmup'] == 120
    for role in ['production', 'portable']:
        assert all(v['cases'] == 400 for v in analysis['reports'][role]['callers'].values())
    assert len(analysis['performance']['controls']) == 63 and len(analysis['performance']['gains']) == 21
    measured = read(PAYLOAD / 'manifests/portable-parakeet.json')
    for field, name in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll')]:
        assert measured[field] == pin(MODEL / 'runtime' / name)['sha256']
    assert pin(MODEL / 'closed.json')['sha256'] == '17e10dce9fdecf5aeb923ab9af0347c02a6850c6d5368bf4d8439eb76d01fc21'
    proof = read(MODEL / 'closed.json'); assert proof['passed']; verify(proof['files'])
    for identity in proof['identities']: terminal(identity)
    assert pin(QUALIFIED / 'closed.json')['sha256'] == 'c231cc61297896a0ded497ff9f7276861f97b872e6ea84e5661e67d587213c6b'
    qualified = read(QUALIFIED / 'closed.json'); assert qualified['passed']; verify(qualified['files'])
    for identity in qualified['identities']: terminal(identity)
    return dict(passed=True, amd_closure=pin(AMD / 'closed.json'), amd_analysis=pin(AMD / 'analysis.json'),
                windows_closure=pin(MODEL / 'closed.json'), suites_closure=pin(QUALIFIED / 'closed.json'), names=NAMES)


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
    for name in NAMES:
        if name.endswith('MathOps.PartialReduction.cs'):
            assert not (ROOT / name).exists()
        else:
            preimage = subprocess.check_output(['git', 'show', '33e35f81:' + name], cwd=ROOT).decode('utf-8-sig').replace('\r\n', '\n')
            assert (ROOT / name).read_text(encoding='utf-8-sig') == preimage, name
    BASE.mkdir(); (BASE / 'logs').mkdir()
    patch = BASE / 'qualified-source.patch'
    patches = []
    for name in NAMES:
        before_text = (ROOT / name).read_text(encoding='utf-8-sig') if (ROOT / name).exists() else ''
        after_text = (QUALIFIED / 'source' / name).read_text(encoding='utf-8-sig')
        patches.extend(difflib.unified_diff(before_text.splitlines(True), after_text.splitlines(True), fromfile='a/' + name, tofile='b/' + name))
    patch.write_text(''.join(patches), encoding='utf8')
    before = {}
    for name in NAMES:
        p = ROOT / name; before[name] = pin(p) if p.exists() else None
        if p.exists():
            target = BASE / 'before' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    accepted.update(before=before, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save(BASE / 'admission.json', accepted)
    for name in NAMES:
        shutil.copy2(QUALIFIED / 'source' / name, ROOT / name)
        assert pin(ROOT / name) == pin(QUALIFIED / 'source' / name)
    bridge = BASE / 'bridge'; bridge.mkdir()
    shutil.copy2(ROOT / 'tests/pyannote/combined-avx512/Inventory.cs.txt', bridge / 'Program.cs')
    shutil.copy2(MODEL / 'bridge/Bridge.csproj', bridge / 'Bridge.csproj')
    consumer = BASE / 'consumer'; consumer.mkdir()
    shutil.copy2(SELECTED / 'consumer/Program.cs', consumer / 'Program.cs')
    shutil.copy2(SELECTED / 'consumer/PackageProbe.csproj', consumer / 'PackageProbe.csproj')
    files = {rel(p): pin(p) for p in [*[ROOT / n for n in NAMES], *TOOLS.glob('*.py'), MONITOR, patch,
              bridge / 'Program.cs', bridge / 'Bridge.csproj', consumer / 'Program.cs', consumer / 'PackageProbe.csproj',
              AMD / 'closed.json', AMD / 'analysis.json', MODEL / 'closed.json', QUALIFIED / 'closed.json', ROOT / 'tests/parakeet/single-panel-amd-v2/admission.py',
              ROOT / 'tests/parakeet/single-panel-amd-v2/candidate_protocol.py']}
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
        assert [(r['assembly'], r['methods']) for r in inventory['observations']] == [('Lokad.Onnx.dll', 3114), ('Lokad.Onnx.Data.dll', 697)]
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
