"""Build only the representation consumer; capture two instruction modes."""
from pathlib import Path
import shutil
import sys
import common

BASE=Path(__file__).resolve().parent
common.BASE=BASE
pin,read,save,verify,idle,live=common.pin,common.read,common.save,common.verify,common.idle,common.live
DOTNET=common.DOTNET


def build(state,env,spec):
    for name in ['runtime','tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    for name,wanted in spec['runtime_files'].items():
        source=Path(spec['original_runtime'])/name
        assert pin(source)==wanted
        shutil.copy2(source,BASE/'runtime'/name)
    env=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',
        DOTNET_CLI_TELEMETRY_OPTOUT='1',NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),
        MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))
    flags=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
        '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:FrozenProductDirectory='+str(BASE/'runtime')]
    limits=spec['build_limits'];project=BASE/'source/PackedLogicalWeightProbe.csproj'
    common.job(state,'sdk-version',[DOTNET,'--version'],env,project.parent,limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    common.job(state,'consumer-restore',[DOTNET,'restore',project,*flags,'--source',spec['feed'],'--packages',BASE/'packages'],env,project.parent,limits,spec)
    common.job(state,'consumer-build',[DOTNET,'build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],env,project.parent,limits,spec)
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        name='PackedLogicalWeightProbe.'+suffix
        shutil.copy2(BASE/'source/bin/Release/net10.0'/name,BASE/'runtime'/name)
    for name,wanted in spec['runtime_files'].items():assert pin(BASE/'runtime'/name)==wanted
    save(BASE/'built.json',dict(passed=True,product_rebuilt=False,runtime={p.name:pin(p) for p in (BASE/'runtime').iterdir()}))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json')
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted
    for mode in ['native','hardware-disabled']:
        environment=dict(env)
        if mode=='hardware-disabled':environment['DOTNET_EnableHWIntrinsic']='0'
        common.job(state,mode,[DOTNET,BASE/'runtime/PackedLogicalWeightProbe.dll',BASE/'spec.json',BASE/'probe'/mode,mode],
            environment,BASE,spec['capture_limits'],spec)
        result=read(BASE/'probe'/mode/'result.json')
        assert result['completed'] and result['no_model_execution'] and result['no_product_change']


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
