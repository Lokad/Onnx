"""Build only the scoped Core/consumer observer, then capture original requests."""
import importlib.util
import os
from pathlib import Path
import shutil
import common
from common import BASE, DOTNET, pin, read, save, live, idle, verify, job


def build(state, env, spec):
    for name in ['tmp','cli-home','packages','http-cache']: (BASE/name).mkdir()
    env = dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
        NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))
    flags = ['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
        '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:FrozenProductDirectory='+spec['prior']]
    limits = spec['build_limits']
    job(state,'sdk-version',[DOTNET,'--version'],env,BASE/'source',limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip() == '10.0.204'
    projects = [('core',BASE/'source/src/Lokad.Onnx/Lokad.Onnx.csproj'),
                ('consumer',BASE/'consumer-source/SampledAudio.csproj'),('bridge',BASE/'bridge-source/Bridge.csproj')]
    for name, path in projects:
        job(state,name+'-restore',[DOTNET,'restore',path,*flags,'--source','/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed','--packages',BASE/'packages'],env,path.parent,limits,spec)
        job(state,name+'-build',[DOTNET,'build',path,'-c','Release',*flags,'--no-restore','--disable-build-servers'],env,path.parent,limits,spec)
    folder = BASE/'runtime-observed'; folder.mkdir()
    for original in Path(spec['prior']).iterdir():
        if not original.is_file(): continue
        source = original
        if original.name.startswith('SampledAudio.'):
            source = BASE/'consumer-source/bin/Release/net10.0'/original.name
        elif original.name == 'Lokad.Onnx.dll':
            source = BASE/'source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll'
        shutil.copy2(source,folder/original.name)
    assert pin(folder/'Lokad.Onnx.Data.dll') == spec['data']
    (BASE/'inventory').mkdir()
    job(state,'inventory',[DOTNET,BASE/'bridge-source/bin/Release/net10.0/Bridge.dll',spec['prior'],folder,BASE/'inventory/instructions.json'],env,BASE,limits,spec)
    save(BASE/'built.json',dict(core=pin(folder/'Lokad.Onnx.dll'),data=pin(folder/'Lokad.Onnx.Data.dll'),
        consumer=pin(folder/'SampledAudio.dll'),runtime_files={p.relative_to(BASE).as_posix():pin(p) for p in folder.iterdir()}))


def capture(state, env, spec):
    review = read(BASE/'build-review.json'); assert review['passed'] and review['built'] == pin(BASE/'built.json')
    built = read(BASE/'built.json')
    for name,wanted in built['runtime_files'].items(): assert pin(BASE/name) == wanted
    app = Path(spec['app']); runtime = BASE/'runtime-observed'; output = BASE/'phase'
    module = importlib.util.spec_from_file_location('cpu_accounting',app/'runtime/campaign_processes.py')
    accounting = importlib.util.module_from_spec(module); module.loader.exec_module(accounting)
    env = {k:v for k,v in env.items() if not k.lower().startswith('parakeet_layout_')}
    env['PARAKEET_LAYOUT_CORE_SHA'] = built['core']['sha256']
    before = accounting.snapshot()
    job(state,'phase',[DOTNET,runtime/'SampledAudio.dll',app/'assets',app/'manifests/current-parakeet.json',output,'timing','control'],env,BASE,spec['capture_limits'],spec,output)
    after = accounting.snapshot(); row = state['runs'][-1]
    row['cpu_before'] = before; row['cpu_after'] = after
    row['accounting'] = accounting.foreign_fraction(before,after,state['supervisor']['pid'])
    assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction'] <= .01
    save(BASE/'capture-state.json',state)
    value = read(output/'result.json'); assert value['passed'] and len(value['records']) == 80


if __name__ == '__main__':
    common.build, common.capture = build, capture
    raise SystemExit(common.main())
