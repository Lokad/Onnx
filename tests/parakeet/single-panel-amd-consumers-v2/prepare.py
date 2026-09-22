"""Build and locally qualify consumers before freezing the fresh AMD campaign."""
import importlib.util
import json
import shutil
import traceback
from adapt import ROOT, TOOLS, CORE, DATA, CONTROL_CORE, caller_source, probe_source, bridge_source

BASE = ROOT / 'artifacts/parakeet-single-panel-amd-consumers-v2-20260922'
PRODUCT = ROOT / 'artifacts/parakeet-single-panel-models-20260922/runtime'
CONTROL = ROOT / 'artifacts/pyannote-single-panel-models-20260922/runtime'
PRIOR = ROOT / 'artifacts/pyannote-single-panel-composition-20260922'
FEED = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
SHAPES = ROOT / 'artifacts/pyannote-single-panel-direct-20260922/shapes.json'
MONITOR = ROOT / 'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('amd_consumer_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)
monitor.BASE = BASE
pin, read, save, verify, terminal, psutil = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal, monitor.psutil


def rel(path):
    return path.relative_to(ROOT).as_posix()


def inspect():
    value = read(BASE / 'caller-instructions.json')
    assert value['inventory_complete'] and len(value['observations']) == 1
    row = value['observations'][0]
    assert row['assembly'] == 'Caller.dll' and row['public_surface_equal']
    assert not row['added'] and not row['removed'] and len(row['differences']) == 1
    key = row['differences'][0]; assert key.startswith('CallerProbe::Main::')
    before = row['normalized_methods'][key]; after = row['candidate_methods'][key]
    old = 'e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838'
    assert before.count(old) == 1 and before.replace(old, CONTROL_CORE) == after
    assert row['unchanged_methods'] == row['methods'] - 1
    assert row['before_sha256'] == pin(PRIOR / 'caller/bin/Release/net10.0/Caller.dll')['sha256']
    assert row['after_sha256'] == pin(BASE / 'caller/bin/Release/net10.0/Caller.dll')['sha256']
    return dict(methods=row['methods'], unchanged=row['unchanged_methods'], changed='Main baseline identity literal only')


def main():
    assert not BASE.exists()
    failed = ROOT / 'artifacts/parakeet-single-panel-amd-consumers-20260922/failure-closed.json'
    assert pin(failed)['sha256'] == '479efe68024f2a91437802fb9c0908960322aa0504fa369bec5fbdc6d9d2b634'
    failure = read(failed); assert failure['passed'] and failure['no_consumer_build'] and failure['no_deployment']
    verify(failure['files'])
    closures = {
        'parakeet-single-panel-models-20260922': 'da3bc6e322e27c43ab3ff3055759c75410771a9bfeebc1f84813d7ed770da7ea',
        'parakeet-single-panel-suites-20260922': 'c231cc61297896a0ded497ff9f7276861f97b872e6ea84e5661e67d587213c6b',
        'pyannote-single-panel-models-20260922': '56907aea97ef3af546336e61ef713c9ce5d8179d847a8363bbabfdb1463b531c'}
    dependencies = [failed]
    for name, sha in closures.items():
        path = ROOT / 'artifacts' / name / 'closed.json'
        assert pin(path)['sha256'] == sha
        proof = read(path); assert proof['passed']; verify(proof['files'])
        identity_key = 'terminal_identities' if name == 'pyannote-single-panel-models-20260922' else 'identities'
        for identity in proof[identity_key]: terminal(identity)
        dependencies.append(path)
    external_path = ROOT / 'artifacts/parakeet-single-panel-composition-20260922/external-operand-proof.json'
    assert pin(external_path)['sha256'] == '3b002a7aa467d05d2f0a4f59a76e57f5b5ff1c4d59b0405f94522110fc26eb91'
    external = read(external_path); assert external['passed'] and external['operand_files'] == 19
    verify(external['files'])
    assert pin(PRODUCT / 'Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(PRODUCT / 'Lokad.Onnx.Data.dll')['sha256'] == DATA
    assert pin(CONTROL / 'Lokad.Onnx.dll')['sha256'] == CONTROL_CORE
    assert SHAPES.is_file()
    BASE.mkdir(); (BASE / 'logs').mkdir()
    for name in ['caller', 'probe', 'bridge']: (BASE / name).mkdir()
    (BASE / 'caller/Program.cs').write_text(caller_source(), encoding='utf8')
    (BASE / 'caller/Caller.csproj').write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>$(FrozenProductDirectory)/Lokad.Onnx.dll</HintPath></Reference></ItemGroup></Project>\n', encoding='utf8')
    (BASE / 'probe/Probe.cs').write_text(probe_source(), encoding='utf8')
    shutil.copy2(ROOT / 'tests/parakeet/reduction-dispatch/Probe.csproj', BASE / 'probe/Probe.csproj')
    (BASE / 'bridge/Program.cs').write_text(bridge_source(), encoding='utf8')
    bridge_project = ROOT / 'artifacts/parakeet-single-panel-composition-20260922/bridge/Bridge.csproj'
    shutil.copy2(bridge_project, BASE / 'bridge/Bridge.csproj')
    files = {rel(p): pin(p) for folder in [TOOLS, BASE, PRODUCT, CONTROL, PRIOR / 'caller/bin/Release/net10.0'] for p in folder.rglob('*') if p.is_file()}
    for p in [*dependencies, external_path, SHAPES, MONITOR, bridge_project,
              PRIOR / 'caller/Program.cs', ROOT / 'tests/parakeet/reduction-dispatch/Probe.cs',
              ROOT / 'tests/parakeet/reduction-dispatch/Probe.csproj',
              ROOT / 'tests/pyannote/combined-avx512/Inventory.cs.txt']:
        files[rel(p)] = pin(p)
    files.update(external['files'])
    save(BASE / 'inputs.json', dict(passed=True, files=files))
    own = psutil.Process(); state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    save(BASE / 'processes.json', state)

    def run(name, command, output, inference=False):
        monitor.worker(state, BASE / 'processes.json', name, command, ROOT, [0], 8, 8, 900, not inference, output)
        print(name, 'passed', flush=True)

    try:
        for name, project_name in [('bridge', 'Bridge'), ('caller', 'Caller'), ('probe', 'Probe')]:
            project = BASE / name / (project_name + '.csproj')
            flags = monitor.FLAGS + ['-p:NuGetAudit=false', '-p:FrozenProductDirectory=' + str(PRODUCT)]
            run(name + '-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages'], None)
            run(name + '-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], project.parent)
        run('caller-instructions', ['dotnet', BASE / 'bridge/bin/Release/net10.0/Bridge.dll',
            PRIOR / 'caller/bin/Release/net10.0', BASE / 'caller/bin/Release/net10.0', BASE / 'caller-instructions.json'], BASE / 'bridge', True)
        inspect()
        original_env = monitor.clean_env
        for mode in ['normal', 'disabled']:
            monitor.clean_env = original_env if mode == 'normal' else lambda: dict(original_env(), DOTNET_EnableHWIntrinsic='0')
            try:
                run('caller-' + mode, ['dotnet', BASE / 'caller/bin/Release/net10.0/Caller.dll', CONTROL, SHAPES,
                    BASE / ('caller-' + mode + '.json'), mode, CORE], BASE / 'caller', True)
                run('probe-' + mode, ['dotnet', BASE / 'probe/bin/Release/net10.0/Probe.dll', ROOT, CONTROL,
                    BASE / ('probe-' + mode + '.json'), mode, '10.0.12'], BASE / 'probe', True)
            finally: monitor.clean_env = original_env
        verify(files)
        save(BASE / 'prepared.json', dict(passed=True, files=files, production_changed=False, performance_qualified=False))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'processes.json', state)


if __name__ == '__main__': main()
