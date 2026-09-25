"""Build the warning-free observer and inspect Core/Data with their proper scopes."""
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
KIND = 'observer-finalize'
JOBS = ['final-data-restore', 'final-data-build', 'final-core-inventory', 'final-data-inventory']


def verify():
    spec = common.verify()
    repair = common.read(BASE / 'observer-finalize-spec.json')
    for name, wanted in repair['files'].items():
        assert common.pin(BASE / name) == wanted, name
    assert common.pin(BASE / 'spec.json') == repair['original_spec']
    assert common.pin(BASE / 'build-state.json') == repair['original_state']
    old = common.read(BASE / 'build-state.json')
    assert old['complete'] and old['code'] == 1
    assert old['runs'][-1]['name'] == 'data-build' and old['runs'][-1]['code'] == 1
    assert not common.live(old['supervisor'])
    assert all(not common.live(dict(pid=int(p), birth=b)) for r in old['runs'] for p, b in r['members'].items())
    assert common.pin(BASE / 'observer-recovery-state.json') == repair['recovery_state']
    assert common.pin(BASE / 'observer-recovery-spec.json') == repair['recovery_spec']
    for name, wanted in common.read(BASE / 'observer-recovery-spec.json')['files'].items():
        assert common.pin(BASE / name) == wanted, name
    failed = common.read(BASE / 'observer-recovery-state.json')
    assert failed['complete'] and failed['code'] == 1 and not common.live(failed['supervisor'])
    assert all(not common.live(dict(pid=int(p), birth=b)) for r in failed['runs'] for p, b in r['members'].items())
    assert common.pin(BASE / repair['core_path']) == repair['core']
    return spec, repair


def build(state, env, spec, repair):
    for name in ['final-tmp', 'final-cli-home', 'final-packages', 'final-http-cache']:
        (BASE / name).mkdir()
    env.update(DOTNET_CLI_HOME=str(BASE / 'final-cli-home'), DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',
        DOTNET_CLI_TELEMETRY_OPTOUT='1', NUGET_PACKAGES=str(BASE / 'final-packages'),
        NUGET_HTTP_CACHE_PATH=str(BASE / 'final-http-cache'), MSBUILDDISABLENODEREUSE='1',
        DOTNET_CLI_USE_MSBUILD_SERVER='0', TMPDIR=str(BASE / 'final-tmp'))
    project = BASE / 'observer-finalize-source/ObserverData.csproj'
    flags = ['--tl:off', '--nologo', '-v', 'minimal', '-p:UseSharedCompilation=false', '-nr:false',
        '-p:NuGetAudit=false', '-p:EnableSourceControlManagerQueries=false', '-p:EnableSourceLink=false',
        '-p:FrozenProductDirectory=' + str(BASE / 'runtime-original')]
    limits = spec['build_limits']
    common.job(state, JOBS[0], [DOTNET, 'restore', project, *flags, '--source', spec['feed'],
        '--packages', BASE / 'final-packages'], env, project.parent, limits, spec)
    common.job(state, JOBS[1], [DOTNET, 'build', project, '-c', 'Release', *flags,
        '--no-restore', '--disable-build-servers'], env, project.parent, limits, spec)
    for name in ['final-runtime-control', 'final-runtime-observed', 'final-runtime-core-only']:
        folder = BASE / name
        folder.mkdir()
        for original in (BASE / 'runtime-original').iterdir():
            source = original
            if original.name == 'Lokad.Onnx.Data.dll' and name != 'final-runtime-core-only':
                source = project.parent / 'bin/Release/net10.0/Lokad.Onnx.Data.dll'
            elif original.name == 'Lokad.Onnx.dll' and name != 'final-runtime-control':
                source = BASE / repair['core_path']
            shutil.copy2(source, folder / original.name)
    common.job(state, JOBS[2], [DOTNET, BASE / 'bridge/Bridge.dll', BASE / 'runtime-original',
        BASE / 'final-runtime-core-only', BASE / 'inventory/core-instructions.json'], env, BASE, limits, spec)
    common.job(state, JOBS[3], [DOTNET, BASE / 'observer-finalize-inspector/Bridge.dll', BASE / 'runtime-original',
        BASE / 'final-runtime-observed', BASE / 'inventory/data-instructions.json'], env, BASE, limits, spec)
    core = common.read(BASE / 'inventory/core-instructions.json')
    data = common.read(BASE / 'inventory/data-instructions.json')
    assert core['inventory_complete'] and data['inventory_complete']
    assert [r['assembly'] for r in core['observations']] == ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']
    assert [r['assembly'] for r in data['observations']] == ['SampledAudio.dll', 'Lokad.Onnx.Data.dll']
    for row in [core['observations'][1], data['observations'][0]]:
        assert not row['differences'] and not row['added'] and not row['removed']
        assert row['before_sha256'] == row['after_sha256']
    assert not (BASE / 'inventory/instructions.json').exists()
    common.save(BASE / 'inventory/instructions.json', dict(inventory_complete=True,
        scope='Unmodified Core and Data rows from the retained scoped inventories',
        components={name:common.pin(BASE / 'inventory' / name) for name in ['core-instructions.json','data-instructions.json']},
        observations=[core['observations'][0], data['observations'][1]]))
    # The previous failed recovery's complete runtime files are retained locally.
    # Replace only its two Data outputs, after both inventories have completed.
    for name, wanted in repair['previous_runtimes'].items():
        assert common.pin(BASE / name) == wanted, name
    for role in ['control', 'observed']:
        target = BASE / ('runtime-' + role) / 'Lokad.Onnx.Data.dll'
        temporary = target.with_suffix('.final.tmp'); assert not temporary.exists()
        shutil.copy2(BASE / ('final-runtime-' + role) / target.name, temporary)
        temporary.replace(target)
        assert all(common.pin(p) == common.pin(BASE / ('runtime-' + role) / p.name)
                   for p in (BASE / ('final-runtime-' + role)).iterdir())
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
