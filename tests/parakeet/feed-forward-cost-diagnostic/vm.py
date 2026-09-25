"""Bounded diagnostic build/capture using the retained VM process supervisor."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys

import common

BASE = Path(__file__).resolve().parent
DOTNET = '/home/vermorel/.dotnet/dotnet'
MODES = ['clock', 'stages', 'markers']
BUILD_JOBS = ['sdk-version', 'core-restore', 'core-build', 'data-restore', 'data-build', 'inventory']
common.BASE = BASE
pin, read, save, verify, idle, live = common.pin, common.read, common.save, common.verify, common.idle, common.live


def build(state, env, spec):
    for name in ['tmp', 'cli-home', 'packages', 'http-cache', 'runtime-original', 'inventory']:
        (BASE / name).mkdir()
    # The old observer directory supplies the already qualified consumer and
    # dependencies. Replace both products with the admitted measured release.
    for original in Path(spec['consumer_runtime']).iterdir():
        if not original.is_file():
            continue
        source = Path(spec['product_runtime']) / original.name if original.name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'] else original
        shutil.copy2(source, BASE / 'runtime-original' / original.name)
    for name, wanted in spec['original_runtime_files'].items():
        assert common.pin(BASE / 'runtime-original' / name) == wanted, name
    env = dict(env, DOTNET_CLI_HOME=str(BASE / 'cli-home'), DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',
        DOTNET_CLI_TELEMETRY_OPTOUT='1', NUGET_PACKAGES=str(BASE / 'packages'),
        NUGET_HTTP_CACHE_PATH=str(BASE / 'http-cache'), MSBUILDDISABLENODEREUSE='1',
        DOTNET_CLI_USE_MSBUILD_SERVER='0', TMPDIR=str(BASE / 'tmp'))
    flags = ['--tl:off', '--nologo', '-v', 'minimal', '-p:UseSharedCompilation=false', '-nr:false',
        '-p:NuGetAudit=false', '-p:EnableSourceControlManagerQueries=false', '-p:EnableSourceLink=false',
        '-p:FrozenProductDirectory=' + str(BASE / 'runtime-original')]
    limits = spec['build_limits']
    common.job(state, 'sdk-version', [DOTNET, '--version'], env, BASE / 'core-source', limits, spec)
    assert (BASE / 'logs/sdk-version.stdout').read_text().strip() == '10.0.204'
    projects = [('core', BASE / 'core-source/src/Lokad.Onnx/Lokad.Onnx.csproj'),
                ('data', BASE / 'data-source/ObserverData.csproj')]
    for name, project in projects:
        common.job(state, name + '-restore', [DOTNET, 'restore', project, *flags, '--source', spec['feed'],
            '--packages', BASE / 'packages'], env, project.parent, limits, spec)
        common.job(state, name + '-build', [DOTNET, 'build', project, '-c', 'Release', *flags,
            '--no-restore', '--disable-build-servers'], env, project.parent, limits, spec)
    for name in ['runtime-control', 'runtime-observed']:
        folder = BASE / name
        folder.mkdir()
        for original in (BASE / 'runtime-original').iterdir():
            source = original
            if original.name == 'Lokad.Onnx.Data.dll':
                source = BASE / 'data-source/bin/Release/net10.0/Lokad.Onnx.Data.dll'
            elif original.name == 'Lokad.Onnx.dll' and name == 'runtime-observed':
                source = BASE / 'core-source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll'
            shutil.copy2(source, folder / original.name)
    common.job(state, 'inventory', [DOTNET, BASE / 'bridge/Bridge.dll', BASE / 'runtime-original',
        BASE / 'runtime-observed', BASE / 'inventory/instructions.json'], env, BASE, limits, spec)
    assert [r['name'] for r in state['runs']] == BUILD_JOBS
    common.save(BASE / 'built.json', dict(passed=True,
        core=common.pin(BASE / 'runtime-observed/Lokad.Onnx.dll'),
        data=common.pin(BASE / 'runtime-observed/Lokad.Onnx.Data.dll'),
        original_core=common.pin(BASE / 'runtime-original/Lokad.Onnx.dll'),
        consumer=common.pin(BASE / 'runtime-observed/SampledAudio.dll'),
        runtime_files={p.relative_to(BASE).as_posix(): common.pin(p)
            for name in ['runtime-original', 'runtime-control', 'runtime-observed']
            for p in (BASE / name).iterdir()}))


def capture(state, env, spec):
    approval = common.read(BASE / 'build-review.json')
    assert approval['passed'] and approval['arithmetic_equivalent'] and approval['source_changes_exact']
    assert approval['built'] == common.pin(BASE / 'built.json')
    built = common.read(BASE / 'built.json')
    for name, wanted in built['runtime_files'].items():
        assert common.pin(BASE / name) == wanted, name
    assert built['consumer'] == spec['consumer'] and built['original_core'] == spec['product']['Lokad.Onnx.dll']
    app = Path(spec['app'])
    loader = importlib.util.spec_from_file_location('cpu_accounting', app / 'runtime/campaign_processes.py')
    accounting = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(accounting)
    for mode in MODES:
        common.verify()
        runtime = BASE / ('runtime-observed' if mode == 'markers' else 'runtime-control')
        output = BASE / mode
        environment = dict(env, PARAKEET_PHASE_MODE='phase', PARAKEET_COST_MODE=mode,
            PARAKEET_PHASE_CORE_SHA=common.pin(runtime / 'Lokad.Onnx.dll')['sha256'],
            PARAKEET_PHASE_DATA_SHA=built['data']['sha256'])
        before = accounting.snapshot()
        common.job(state, mode, [DOTNET, runtime / 'SampledAudio.dll', app / 'assets',
            app / spec['manifest'], output, 'timing', 'control'], environment, BASE,
            spec['capture_limits'], spec, output)
        after = accounting.snapshot()
        row = state['runs'][-1]
        row.update(cpu_before=before, cpu_after=after,
            accounting=accounting.foreign_fraction(before, after, state['supervisor']['pid']))
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction'] <= .01
        common.save(BASE / 'capture-state.json', state)
        result = common.read(output / 'result.json')
        assert result['passed'] and len(result['records']) == 80
    assert [r['name'] for r in state['runs']] == MODES


if __name__ == '__main__':
    assert sys.platform == 'linux' and not sys.flags.optimize
    os.environ.pop('PARAKEET_COST_MODE', None)
    common.BASE = BASE
    common.build, common.capture = build, capture
    raise SystemExit(common.main())
