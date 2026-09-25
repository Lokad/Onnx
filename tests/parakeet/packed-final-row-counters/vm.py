"""Build one binary-reference consumer and count four actual encoder corpora."""
from pathlib import Path
import os
import shutil
import sys
import time
import psutil
import common

BASE=Path(__file__).resolve().parent;common.BASE=BASE
pin,read,save,verify,idle,live=common.pin,common.read,common.save,common.verify,common.idle,common.live
DOTNET=common.DOTNET


def runtime_pins():return {p.relative_to(BASE/'runtime').as_posix():pin(p) for p in (BASE/'runtime').rglob('*') if p.is_file()}


def build(state,env,spec):
    receipt=read(spec['model_collection']);assert receipt['terminal'] and receipt['code']==0
    assert not any(live(i) for i in receipt['identities'])
    for name in ['runtime','tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    for role,files in spec['runtime_files'].items():
        (BASE/'runtime'/role).mkdir()
        for name,wanted in files.items():
            source=Path(spec['original_runtime'])/role/name;assert pin(source)==wanted
            os.link(source,BASE/'runtime'/role/name)
    environment=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',
        DOTNET_CLI_TELEMETRY_OPTOUT='1',NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),
        MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))
    flags=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
        '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:FrozenProductDirectory='+str(BASE/'runtime/candidate')]
    limits=spec['build_limits'];project=BASE/'source/OwnedWeightCounters.csproj'
    common.job(state,'sdk-version',[DOTNET,'--version'],environment,project.parent,limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    common.job(state,'consumer-restore',[DOTNET,'restore',project,*flags,'--source',spec['feed'],'--packages',BASE/'packages'],environment,project.parent,limits,spec)
    common.job(state,'consumer-build',[DOTNET,'build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],environment,project.parent,limits,spec)
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        name='OwnedWeightCounters.'+suffix
        shutil.copy2(BASE/'source/bin/Release/net10.0'/name,BASE/'runtime/candidate'/name)
        os.link(BASE/'runtime/candidate'/name,BASE/'runtime/selected'/name)
    for role,files in spec['runtime_files'].items():
        for name,wanted in files.items():assert pin(BASE/'runtime'/role/name)==wanted
    save(BASE/'built.json',dict(passed=True,product_rebuilt=False,runtime=runtime_pins()))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json') and runtime_pins()==built['runtime']
    (BASE/'probe').mkdir()
    for name in spec['jobs']:
        role,mode=name.split('-');verify();assert runtime_pins()==built['runtime']
        # The full-model protocol already permits this wait. Keep every observation.
        started=time.monotonic();observations=[];limits=spec['capture_limits']
        while True:
            value=dict(seconds=time.monotonic()-started,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(BASE).free)
            observations.append(value);save(BASE/(name+'-preflight-wait.json'),observations)
            assert value['seconds']<spec['preflight_wait_seconds'] and value['tmpfs']>=limits['tmpfs_before']
            if value['available']>=limits['available_before']:break
            time.sleep(10)
        environment=dict(env)
        if mode=='256':environment['DOTNET_EnableAVX512']='0'
        output=BASE/'probe'/name
        common.job(state,name,[DOTNET,BASE/'runtime'/role/'OwnedWeightCounters.dll',BASE/'spec.json',output,role,mode],environment,BASE,spec['capture_limits'],spec)
        result=read(output/'result.json');assert result['passed'] and len(result['records'])==20
        assert not result['application_scored'] and not result['forced_gc'] and runtime_pins()==built['runtime']


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
