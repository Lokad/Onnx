"""Serial common-consumer build and four bounded ordinary-runtime screens."""
from pathlib import Path
import shutil
import sys
import common
import campaign_processes as accounting

BASE=Path(__file__).resolve().parent;common.BASE=BASE
pin,read,save,verify,idle,live=common.pin,common.read,common.save,common.verify,common.idle,common.live
DOTNET=common.DOTNET
FLAGS=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
       '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']


def build(state,env,spec):
    for name in ['tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    env=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
        NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',
        DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))
    limits=spec['build_limits'];source=BASE/'source';project=source/'Screen.csproj'
    flags=[*FLAGS,'-p:FrozenProductDirectory='+spec['runtimes']['current']]
    common.job(state,'sdk-version',[DOTNET,'--version'],env,source,limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    common.job(state,'consumer-restore',[DOTNET,'restore',project,*flags,'--source',spec['feed'],'--packages',BASE/'packages'],env,source,limits,spec)
    common.job(state,'consumer-build',[DOTNET,'build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],env,source,limits,spec)
    for role in ['current','candidate']:
        target=BASE/'runtime'/role;shutil.copytree(source/'bin/Release/net10.0',target)
        shutil.copy2(Path(spec['runtimes'][role])/'Lokad.Onnx.dll',target/'Lokad.Onnx.dll')
        assert pin(target/'Lokad.Onnx.dll')==spec['products'][role]
    files={p.relative_to(BASE).as_posix():pin(p) for p in (BASE/'runtime').rglob('*') if p.is_file()}
    save(BASE/'built.json',dict(passed=True,runtime_files=files,consumer=pin(BASE/'runtime/current/Screen.dll'),products=spec['products']))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json')
    for sequence,role in enumerate(['current','candidate','candidate','current']):
        verify()
        for name,wanted in built['runtime_files'].items():assert pin(BASE/name)==wanted,name
        name=role+'-'+str(sequence);runtime=BASE/'runtime'/role
        before=accounting.snapshot()
        common.job(state,name,[DOTNET,runtime/'Screen.dll',BASE,role,sequence,BASE/'logs'/(name+'.json')],
                   env,BASE,spec['capture_limits'],spec)
        after=accounting.snapshot();row=state['runs'][-1]
        row['cpu_before']=before;row['cpu_after']=after
        row['accounting']=accounting.foreign_fraction(before,after,state['supervisor']['pid'])
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction']<=.01
        save(BASE/(state['kind']+'-state.json'),state)
        result=read(BASE/'logs'/(name+'.json'));assert result['passed'] and result['sequence']==sequence
        for path,wanted in built['runtime_files'].items():assert pin(BASE/path)==wanted,path


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
