"""Build the flag-check adapter, then observe the unchanged failed candidate."""
from pathlib import Path
import shutil
import sys
import common

BASE=Path(__file__).resolve().parent;common.BASE=BASE
pin,read,save,verify,idle,live=common.pin,common.read,common.save,common.verify,common.idle,common.live
DOTNET=common.DOTNET


def build(state,env,spec):
    for name in ['tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    env=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
        NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',
        DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))
    flags=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
        '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:FrozenProductDirectory='+spec['prior']]
    project=BASE/'contract-source/TailContracts.csproj';limits=spec['build_limits']
    common.job(state,'sdk-version',[DOTNET,'--version'],env,project.parent,limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    common.job(state,'contract-restore',[DOTNET,'restore',project,*flags,'--source',spec['feed'],'--packages',BASE/'packages'],env,project.parent,limits,spec)
    common.job(state,'contract-build',[DOTNET,'build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],env,project.parent,limits,spec)
    folder=BASE/'runtime';folder.mkdir()
    for name,wanted in spec['runtime'].items():
        source=Path(spec['candidate'])/name;assert pin(source)==wanted;shutil.copy2(source,folder/name)
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        name='TailContracts.'+suffix;shutil.copy2(project.parent/'bin/Release/net10.0'/name,folder/name)
    save(BASE/'built.json',dict(runtime={p.name:pin(p) for p in folder.iterdir()},no_product_build=True))


def capture(state,env,spec):
    build_state=read(BASE/'build-state.json')
    assert build_state['complete'] and build_state['code']==0 and not live(build_state['supervisor'])
    built=read(BASE/'built.json')
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted
    environment=dict(env,DOTNET_EnableAVX512='0',DOTNET_JitDisasm=spec['disasm'])
    common.job(state,'avx512-disabled',[DOTNET,BASE/'runtime/TailContracts.dll',BASE/'spec.json',BASE/'original-built.json',
        BASE/'probe/avx512-disabled','avx512-disabled'],environment,BASE,spec['capture_limits'],spec)
    assert read(BASE/'probe/avx512-disabled/result.json')['completed']
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
