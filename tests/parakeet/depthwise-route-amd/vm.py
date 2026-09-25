"""Build an observer Core, retain exact Data/consumer, capture once with bounds."""
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
    limits=spec['build_limits']
    common.job(state,'sdk-version',[DOTNET,'--version'],env,BASE/'source',limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name,path in [('core',BASE/'source/src/Lokad.Onnx/Lokad.Onnx.csproj'),('bridge',BASE/'bridge-source/Bridge.csproj')]:
        common.job(state,name+'-restore',[DOTNET,'restore',path,*FLAGS,'--source',spec['feed'],'--packages',BASE/'packages'],env,path.parent,limits,spec)
        common.job(state,name+'-build',[DOTNET,'build',path,'-c','Release',*FLAGS,'--no-restore','--disable-build-servers'],env,path.parent,limits,spec)
    runtime=BASE/'runtime';runtime.mkdir()
    for name,wanted in spec['runtime'].items():
        original=Path(spec['prior'])/name;assert pin(original)==wanted
        source=BASE/'source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll' if name=='Lokad.Onnx.dll' else original
        shutil.copy2(source,runtime/name)
    common.job(state,'inventory',[DOTNET,BASE/'bridge-source/bin/Release/net10.0/Bridge.dll',spec['prior'],runtime,
        BASE/'logs/instructions.json'],env,BASE,limits,spec)
    files={p.name:pin(p) for p in runtime.iterdir() if p.is_file()}
    save(BASE/'built.json',dict(passed=True,runtime=files,product={n:files[n] for n in spec['before_product']},
        consumer=files['SampledAudio.dll'],inventory=pin(BASE/'logs/instructions.json')))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json')
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted,name
    app=Path(spec['app']);output=BASE/'probe'
    environment=dict(env,PARAKEET_PHASE_MODE='control',PARAKEET_PHASE_CORE_SHA=built['product']['Lokad.Onnx.dll']['sha256'],
        PARAKEET_PHASE_DATA_SHA=built['product']['Lokad.Onnx.Data.dll']['sha256'],PARAKEET_DEPTHWISE_COUNTS=str(BASE/'logs/counts.json'))
    before=accounting.snapshot()
    common.job(state,'counts',[DOTNET,BASE/'runtime/SampledAudio.dll',app/'assets',app/'manifests/current-parakeet.json',output,'timing','control'],
        environment,BASE,spec['capture_limits'],spec,output)
    after=accounting.snapshot();row=state['runs'][-1]
    row['cpu_before']=before;row['cpu_after']=after;row['accounting']=accounting.foreign_fraction(before,after,state['supervisor']['pid'])
    assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction']<=.01
    save(BASE/(state['kind']+'-state.json'),state)
    assert read(output/'result.json')['passed'] and read(BASE/'logs/counts.json')['passed']
    for name,wanted in built['runtime'].items():assert pin(BASE/'runtime'/name)==wanted,name


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
