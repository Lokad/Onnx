"""Build only the arithmetic consumer, preserving both original Core binaries."""
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
    for role,files in spec['runtimes'].items():
        folder=BASE/'runtime'/role;folder.mkdir(parents=True)
        for name,wanted in files.items():
            source=Path(spec['origins'][role])/name;assert pin(source)==wanted;shutil.copy2(source,folder/name)
        for suffix in ['dll','deps.json','runtimeconfig.json']:
            name='TailContracts.'+suffix;shutil.copy2(project.parent/'bin/Release/net10.0'/name,folder/name)
    save(BASE/'built.json',dict(products=spec['products'],runtime={p.relative_to(BASE/'runtime').as_posix():pin(p) for p in (BASE/'runtime').rglob('*') if p.is_file()}))
    save(BASE/'built-baseline.json',dict(products=dict(baseline=spec['products']['baseline'],candidate=spec['products']['baseline'])))


def capture(state,env,spec):
    state_build=read(BASE/'build-state.json');assert state_build['complete'] and state_build['code']==0 and not live(state_build['supervisor'])
    built=read(BASE/'built.json')
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted
    for job in spec['jobs']:
        environment=dict(env);mode=job['mode'];role=job['role']
        if mode.startswith('scalar-'):environment['DOTNET_EnableHWIntrinsic']='0'
        else:
            environment['DOTNET_JitDisasm']=spec['disasm']
            if mode=='avx512-disabled':environment['DOTNET_EnableAVX512']='0'
        built_file='built-baseline.json' if role=='baseline' else 'built.json'
        common.job(state,job['name'],[DOTNET,BASE/'runtime'/role/'TailContracts.dll',BASE/'spec.json',BASE/built_file,
            BASE/'probe'/job['name'],mode],environment,BASE,spec['capture_limits'],spec)
        assert read(BASE/'probe'/job['name']/'result.json')['completed']
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
