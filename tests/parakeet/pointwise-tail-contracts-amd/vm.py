"""Bounded single-candidate build and contracts; no model or timing trial."""
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
    limits=spec['build_limits']
    common.job(state,'sdk-version',[DOTNET,'--version'],env,BASE/'source',limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name,path in [('core',BASE/'source/src/Lokad.Onnx/Lokad.Onnx.csproj'),('contract',BASE/'contract-source/TailContracts.csproj'),('bridge',BASE/'bridge-source/Bridge.csproj')]:
        common.job(state,name+'-restore',[DOTNET,'restore',path,*flags,'--source',spec['feed'],'--packages',BASE/'packages'],env,path.parent,limits,spec)
        common.job(state,name+'-build',[DOTNET,'build',path,'-c','Release',*flags,'--no-restore','--disable-build-servers'],env,path.parent,limits,spec)
    # Both runtimes are below runtime/ so the unchanged collector retains them.
    for role in ['baseline','candidate']:
        folder=BASE/'runtime'/role;folder.mkdir(parents=True)
        for name,wanted in spec['runtime'].items():
            original=Path(spec['prior'])/name;assert pin(original)==wanted
            source=BASE/'source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll' if role=='candidate' and name=='Lokad.Onnx.dll' else original
            shutil.copy2(source,folder/name)
        for suffix in ['dll','deps.json','runtimeconfig.json']:
            name='TailContracts.'+suffix;shutil.copy2(BASE/'contract-source/bin/Release/net10.0'/name,folder/name)
    common.job(state,'inventory',[DOTNET,BASE/'bridge-source/bin/Release/net10.0/Bridge.dll',BASE/'runtime/baseline',
        BASE/'runtime/candidate',BASE/'logs/instructions.json',BASE/'runtime/baseline'],env,BASE,limits,spec)
    runtime={p.relative_to(BASE/'runtime').as_posix():pin(p) for p in (BASE/'runtime').rglob('*') if p.is_file()}
    products={role:{n:runtime[role+'/'+n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']} for role in ['baseline','candidate']}
    save(BASE/'built.json',dict(passed=True,runtime=runtime,products=products,inventory=pin(BASE/'logs/instructions.json')))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json')
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted
    for mode in spec['modes']:
        environment=dict(env)
        if mode=='normal':environment['DOTNET_JitDisasm']=spec['disasm']
        elif mode=='avx512-disabled':environment['DOTNET_EnableAVX512']='0'
        else:environment['DOTNET_EnableHWIntrinsic']='0'
        role='baseline' if mode=='scalar-baseline' else 'candidate'
        common.job(state,mode,[DOTNET,BASE/'runtime'/role/'TailContracts.dll',BASE/'spec.json',BASE/'built.json',BASE/'probe'/mode,mode],
            environment,BASE,spec['capture_limits'],spec)
        assert read(BASE/'probe'/mode/'result.json')['completed']
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
