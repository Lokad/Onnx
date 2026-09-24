"""Reuse the bounded phase supervisor; rebuild only its consumer identity check."""
import importlib.util
from pathlib import Path
import shutil

import common
from common import BASE, DOTNET, idle, job, live, pin, read, save, verify


def build(state, env, spec):
    for name in ['tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    env=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',
        DOTNET_CLI_TELEMETRY_OPTOUT='1',NUGET_PACKAGES=str(BASE/'packages'),
        NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',
        DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))
    flags=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false',
        '-p:NuGetAudit=false','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false',
        '-p:FrozenProductDirectory='+spec['prior']]
    limits=spec['build_limits']
    job(state,'sdk-version',[DOTNET,'--version'],env,BASE/'consumer-source',limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name,project in [('consumer','SampledAudio'),('bridge','Bridge')]:
        folder=BASE/(name+'-source');path=folder/(project+'.csproj')
        job(state,name+'-restore',[DOTNET,'restore',path,*flags,'--source',
            '/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
            '--packages',BASE/'packages'],env,folder,limits,spec)
        job(state,name+'-build',[DOTNET,'build',path,'-c','Release',*flags,
            '--no-restore','--disable-build-servers'],env,folder,limits,spec)
    for name in ['runtime-control','runtime-observed']:
        folder=BASE/name;folder.mkdir()
        for original in Path(spec['prior']).iterdir():
            if not original.is_file():continue
            source=original
            if original.name.startswith('SampledAudio.'):
                source=BASE/'consumer-source/bin/Release/net10.0'/original.name
            elif original.name=='Lokad.Onnx.dll' and name=='runtime-observed':
                source=BASE/'candidate/Lokad.Onnx.dll'
            shutil.copy2(source,folder/original.name)
        assert pin(folder/'Lokad.Onnx.dll')==spec['core' if name=='runtime-control' else 'candidate_core']
        assert pin(folder/'Lokad.Onnx.Data.dll')==spec['data']
    (BASE/'inventory').mkdir()
    job(state,'inventory',[DOTNET,BASE/'bridge-source/bin/Release/net10.0/Bridge.dll',
        spec['prior'],BASE/'runtime-observed',BASE/'inventory/instructions.json'],env,BASE,limits,spec)
    save(BASE/'built.json',dict(core=pin(BASE/'runtime-control/Lokad.Onnx.dll'),
        candidate_core=pin(BASE/'runtime-observed/Lokad.Onnx.dll'),data=spec['data'],
        consumer=pin(BASE/'runtime-observed/SampledAudio.dll'),
        runtime_files={p.relative_to(BASE).as_posix():pin(p)
            for folder in ['runtime-control','runtime-observed'] for p in (BASE/folder).iterdir()}))


def capture(state,env,spec):
    approval=read(BASE/'build-review.json');assert approval['passed'] and approval['built']==pin(BASE/'built.json')
    built=read(BASE/'built.json')
    for name,wanted in built['runtime_files'].items():assert pin(BASE/name)==wanted
    app=Path(spec['app'])
    module=importlib.util.spec_from_file_location('cpu_accounting',app/'runtime/campaign_processes.py')
    accounting=importlib.util.module_from_spec(module);module.loader.exec_module(accounting)
    for name in ['control','wall']:
        verify();runtime=BASE/('runtime-control' if name=='control' else 'runtime-observed');output=BASE/name
        environment=dict(env,PARAKEET_PHASE_MODE='wall',PARAKEET_PHASE_DATA_SHA=spec['data']['sha256'],
            PARAKEET_PHASE_CORE_SHA=spec['core' if name=='control' else 'candidate_core']['sha256'])
        before=accounting.snapshot()
        job(state,name,[DOTNET,runtime/'SampledAudio.dll',app/'assets',app/'manifests/current-parakeet.json',
            output,'timing','control'],environment,BASE,spec['capture_limits'],spec,output)
        after=accounting.snapshot();row=state['runs'][-1]
        row.update(cpu_before=before,cpu_after=after,
            accounting=accounting.foreign_fraction(before,after,state['supervisor']['pid']))
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction']<=.01
        save(BASE/'capture-state.json',state)
        value=read(output/'result.json');assert value['passed'] and len(value['records'])==80


if __name__=='__main__':
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
