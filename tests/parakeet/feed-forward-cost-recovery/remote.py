"""Complete only the corrected observer build and missing inventory."""
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import psutil
import common

BASE = Path(__file__).resolve().parent
common.BASE = BASE
DOTNET = '/home/vermorel/.dotnet/dotnet'
KIND = 'observer-recovery'
JOBS = ['observer-data-restore', 'observer-data-build', 'observer-inventory']


def verify():
    spec = common.verify()
    repair = common.read(BASE / 'observer-recovery-spec.json')
    for name, wanted in repair['files'].items():
        assert common.pin(BASE / name) == wanted, name
    assert common.pin(BASE / 'spec.json') == repair['original_spec']
    assert common.pin(BASE / 'build-state.json') == repair['original_state']
    old = common.read(BASE / 'build-state.json')
    assert old['complete'] and old['code'] == 1
    assert old['runs'][-1]['name'] == 'data-build' and old['runs'][-1]['code'] == 1
    assert not common.live(old['supervisor'])
    assert all(not common.live(dict(pid=int(p), birth=b)) for r in old['runs'] for p, b in r['members'].items())
    assert common.pin(BASE / repair['core_path']) == repair['core']
    return spec, repair


def build(state, env, spec, repair):
    for name in ['observer-tmp', 'observer-cli-home', 'observer-packages', 'observer-http-cache']:
        (BASE / name).mkdir()
    env.update(DOTNET_CLI_HOME=str(BASE / 'observer-cli-home'), DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',
        DOTNET_CLI_TELEMETRY_OPTOUT='1', NUGET_PACKAGES=str(BASE / 'observer-packages'),
        NUGET_HTTP_CACHE_PATH=str(BASE / 'observer-http-cache'), MSBUILDDISABLENODEREUSE='1',
        DOTNET_CLI_USE_MSBUILD_SERVER='0', TMPDIR=str(BASE / 'observer-tmp'))
    project = BASE / 'observer-recovery-source/ObserverData.csproj'
    flags = ['--tl:off', '--nologo', '-v', 'minimal', '-p:UseSharedCompilation=false', '-nr:false',
        '-p:NuGetAudit=false', '-p:EnableSourceControlManagerQueries=false', '-p:EnableSourceLink=false',
        '-p:FrozenProductDirectory=' + str(BASE / 'runtime-original')]
    limits = spec['build_limits']
    common.job(state, JOBS[0], [DOTNET, 'restore', project, *flags, '--source', spec['feed'],
        '--packages', BASE / 'observer-packages'], env, project.parent, limits, spec)
    common.job(state, JOBS[1], [DOTNET, 'build', project, '-c', 'Release', *flags,
        '--no-restore', '--disable-build-servers'], env, project.parent, limits, spec)
    for name in ['runtime-control', 'runtime-observed']:
        folder = BASE / name
        folder.mkdir()
        for original in (BASE / 'runtime-original').iterdir():
            source = original
            if original.name == 'Lokad.Onnx.Data.dll':
                source = project.parent / 'bin/Release/net10.0/Lokad.Onnx.Data.dll'
            elif original.name == 'Lokad.Onnx.dll' and name == 'runtime-observed':
                source = BASE / repair['core_path']
            shutil.copy2(source, folder / original.name)
    common.job(state, JOBS[2], [DOTNET, BASE / 'bridge/Bridge.dll', BASE / 'runtime-original',
        BASE / 'runtime-observed', BASE / 'inventory/instructions.json'], env, BASE, limits, spec)
    assert [r['name'] for r in state['runs']] == JOBS
    assert not (BASE / 'built.json').exists()
    common.save(BASE / 'built.json', dict(passed=True,
        core=common.pin(BASE / 'runtime-observed/Lokad.Onnx.dll'),
        data=common.pin(BASE / 'runtime-observed/Lokad.Onnx.Data.dll'),
        original_core=common.pin(BASE / 'runtime-original/Lokad.Onnx.dll'),
        consumer=common.pin(BASE / 'runtime-observed/SampledAudio.dll'),
        runtime_files={p.relative_to(BASE).as_posix(): common.pin(p)
            for name in ['runtime-original', 'runtime-control', 'runtime-observed']
            for p in (BASE / name).iterdir()}))
    assert common.pin(BASE / 'runtime-observed/Lokad.Onnx.dll') == repair['core']


def main():
    assert sys.platform == 'linux' and not sys.flags.optimize
    own = psutil.Process(); own.cpu_affinity([0]); common.idle()
    spec, repair = verify()
    path = BASE / (KIND + '-state.json'); assert not path.exists()
    state = dict(kind=KIND, complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()),
                 runs=[], started=time.time())
    common.save(path, state)
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_', 'parakeet_'))}
    env.pop('PYTHONOPTIMIZE', None)
    env['PATH'] = str(Path(DOTNET).parent) + os.pathsep + env.get('PATH', '')
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = '1'
    try:
        build(state, env, spec, repair)
        verify(); state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True, ended=time.time()); common.save(path, state)
    return state['code']


if __name__ == '__main__':
    raise SystemExit(main())
