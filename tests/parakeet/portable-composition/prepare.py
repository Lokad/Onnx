"""Compose the qualified numerical fix with normal portable pyannote source."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-portable-composition-20260922'
PORTABLE = ROOT / 'artifacts/pyannote-portable-integration-tests-20260922'
CONTROL = ROOT / 'artifacts/pyannote-portable-applications-20260922/runtime'
ARITHMETIC = ROOT / 'artifacts/parakeet-reduction-dispatch-20260921/source/src/Lokad.Onnx'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('portable_arithmetic_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()


def inspect():
    value = read(BASE / 'instructions.json')
    assert value['inventory_complete']
    assert [(r['assembly'], r['methods']) for r in value['observations']] == [
        ('Lokad.Onnx.dll', 3108), ('Lokad.Onnx.Data.dll', 697)]
    for row in value['observations']:
        assert row['public_surface_equal'] and not row['removed']
        assert row['before_sha256'] == pin(CONTROL / row['assembly'])['sha256']
        assert row['after_sha256'] == pin(BASE / 'runtime' / row['assembly'])['sha256']
        if row['assembly'] == 'Lokad.Onnx.dll':
            assert sorted(k.split('::')[1] for k in row['differences']) == ['RunFloatMatMulKernel', 'RunPreparedPackedRows']
            assert len(row['added']) == 1 and row['added'][0].startswith('Lokad.Onnx.MathOps::TryPackedPartialSums::')
            assert row['unchanged_methods'] == 3106
        else:
            assert not row['added'] and not row['differences'] and row['unchanged_methods'] == 697
    return value


def main():
    assert not BASE.exists()
    assert pin(PORTABLE / 'closed.json')['sha256'] == 'bf9822426cd75a86cc8c02c065005f0005e1a80b4ee63d5c53313fed145235f6'
    proof = read(PORTABLE / 'closed.json')
    assert proof['passed']
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    assert pin(CONTROL / 'Lokad.Onnx.dll')['sha256'] == 'e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838'
    assert pin(CONTROL / 'Lokad.Onnx.Data.dll')['sha256'] == '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'
    wanted = {'TensorOps.MatMul.cs': 'f0a17a16848324b5be3b465854103aa3d539df228c93896e2b2afac01dcafbc9',
        'MathOps.PartialReduction.cs': '94617be5d0ad04ac1c1782eedfa55a45200e5c74470bb5be68105dc3037a99c9'}
    for name, sha in wanted.items():
        assert pin(ARITHMETIC / name)['sha256'] == sha
    preimage = subprocess.check_output(['git', 'show', 'b0f3ff10:src/Lokad.Onnx/TensorOps.MatMul.cs'], cwd=ROOT).decode('utf-8-sig').replace('\r\n', '\n')
    assert (PORTABLE / 'source/src/Lokad.Onnx/TensorOps.MatMul.cs').read_text(encoding='utf-8-sig') == preimage
    BASE.mkdir(); (BASE / 'logs').mkdir()
    source = BASE / 'source'
    shutil.copytree(PORTABLE / 'source', source, ignore=shutil.ignore_patterns('bin', 'obj'))
    for name in wanted:
        shutil.copy2(ARITHMETIC / name, source / 'src/Lokad.Onnx' / name)
    bridge, probe = BASE / 'bridge', BASE / 'probe'
    bridge.mkdir(); probe.mkdir()
    shutil.copy2(ROOT / 'tests/pyannote/combined-avx512/Inventory.cs.txt', bridge / 'Program.cs')
    shutil.copy2(ROOT / 'artifacts/pyannote-portable-integration-20260922/bridge/Bridge.csproj', bridge / 'Bridge.csproj')
    for name in ('Probe.cs', 'Probe.csproj'):
        shutil.copy2(ROOT / 'tests/parakeet/reduction-dispatch' / name, probe / name)
    files = {rel(p): pin(p) for folder in (source, bridge, probe, TOOLS) for p in folder.rglob('*') if p.is_file()}
    for p in [MONITOR, PORTABLE / 'closed.json', *[ARITHMETIC / n for n in wanted], *CONTROL.glob('*.dll')]:
        files[rel(p)] = pin(p)
    save(BASE / 'inputs.json', dict(passed=True, files=files, product_changes=list(wanted), production_changed=False))
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']

    def run(name, command, output):
        monitor.worker(state, BASE / 'processes.json', name, command, ROOT, [0], 8, 8, 900, True, output)
        print(name, 'passed', flush=True)

    def build(name, project, extra):
        run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages', *extra], None)
        run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', *extra], project.parent)

    try:
        build('cli', source / 'src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj', [])
        shutil.copytree(source / 'src/Lokad.Onnx.CLI/bin/Release/net10.0', BASE / 'runtime')
        build('bridge', bridge / 'Bridge.csproj', [])
        run('instructions', ['dotnet', bridge / 'bin/Release/net10.0/Bridge.dll', CONTROL, BASE / 'runtime', BASE / 'instructions.json'], bridge)
        inspect()
        build('probe', probe / 'Probe.csproj', ['-p:FrozenProductDirectory=' + str(BASE / 'runtime')])
        binary = probe / 'bin/Release/net10.0/Probe.dll'
        run('geometry', ['dotnet', binary, ROOT, CONTROL, BASE / 'geometry.json', 'normal'], probe)
        original_env = monitor.clean_env
        monitor.clean_env = lambda: dict(original_env(), DOTNET_EnableHWIntrinsic='0')
        try:
            run('hardware-off', ['dotnet', binary, ROOT, CONTROL, BASE / 'hardware-off.json', 'disabled'], probe)
        finally:
            monitor.clean_env = original_env
        assert read(BASE / 'geometry.json')['passed'] and read(BASE / 'hardware-off.json')['passed']
        verify(files)
        save(BASE / 'prepared.json', dict(passed=True, files=files, core=pin(BASE / 'runtime/Lokad.Onnx.dll'),
            data=pin(BASE / 'runtime/Lokad.Onnx.Data.dll'), instructions=pin(BASE / 'instructions.json'),
            geometry=pin(BASE / 'geometry.json'), hardware_off=pin(BASE / 'hardware-off.json'),
            model_qualified=False, performance_qualified=False, production_changed=False))
        state['code'] = 0
        print(json.dumps(dict(core=pin(BASE / 'runtime/Lokad.Onnx.dll'), data=pin(BASE / 'runtime/Lokad.Onnx.Data.dll'))), flush=True)
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


if __name__ == '__main__':
    main()
