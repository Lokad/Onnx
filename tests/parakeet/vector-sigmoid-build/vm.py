"""Use the established bounded build supervisor, then run only focused contracts."""
from pathlib import Path
import shutil
import sys
import common

BASE=Path(__file__).resolve().parent;common.BASE=BASE
pin,read,save,verify,idle,live=common.pin,common.read,common.save,common.verify,common.idle,common.live
DOTNET=common.DOTNET
FLAGS=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
       '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']


def environment(env):
    return dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
        NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',
        DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))


def build(state,env,spec):
    for name in ['tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    env=environment(env);limits=spec['build_limits']
    common.job(state,'sdk-version',[DOTNET,'--version'],env,BASE/'source',limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    projects=[('backend',BASE/'source/tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
        ('bridge',BASE/'bridge-source/Bridge.csproj')]
    for name,path in projects:
        common.job(state,name+'-restore',[DOTNET,'restore',path,*FLAGS,'--source',spec['feed'],'--packages',BASE/'packages'],env,path.parent,limits,spec)
        common.job(state,name+'-build',[DOTNET,'build',path,'-c','Release',*FLAGS,'--no-restore','--disable-build-servers'],env,path.parent,limits,spec)
    test_bin=projects[0][1].parent/'bin/Release/net10.0'
    shutil.copytree(test_bin,BASE/'runtime')
    common.job(state,'inventory',[DOTNET,BASE/'bridge-source/bin/Release/net10.0/Bridge.dll',spec['prior'],BASE/'runtime',
        BASE/'logs/instructions.json'],env,BASE,limits,spec)
    runtime={p.relative_to(BASE/'runtime').as_posix():pin(p) for p in (BASE/'runtime').rglob('*') if p.is_file()}
    save(BASE/'built.json',dict(passed=True,product={n:runtime[n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},
        consumer=runtime['Lokad.Onnx.Backend.Tests.dll'],runtime=runtime,test_bin=str(test_bin),inventory=pin(BASE/'logs/instructions.json')))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json')
    env=environment(env);project=BASE/'source/tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
    selected='FullyQualifiedName~SigmoidVectorTests|FullyQualifiedName~CpuExecutionProviderEncoderGapTests.Sigmoid|FullyQualifiedName~EncoderFoundationEdgeTests.UnaryOperatorsPreserveSpecialValuesAndHandleEmptyAndScalarShapes'
    for mode in ['normal','scalar']:
        for name,wanted in built['runtime'].items():
            assert pin(BASE/'runtime'/name)==wanted and pin(Path(built['test_bin'])/name)==wanted,name
        run_env=dict(env,VECTOR_SIGMOID_MODE=mode,VECTOR_SIGMOID_CORE_SHA=built['product']['Lokad.Onnx.dll']['sha256'],
            VECTOR_SIGMOID_DATA_SHA=built['product']['Lokad.Onnx.Data.dll']['sha256'],VECTOR_SIGMOID_TEST_SHA=built['consumer']['sha256'],
            VECTOR_SIGMOID_EVIDENCE=str(BASE/'logs'))
        if mode=='scalar':run_env['DOTNET_EnableHWIntrinsic']='0'
        command=[DOTNET,'test',project,'-c','Release',*FLAGS,'--no-build','--no-restore','--filter',selected,
            '--logger','trx;LogFileName=sigmoid-'+mode+'.trx','--results-directory',BASE/'logs']
        common.job(state,'contracts-'+mode,command,run_env,BASE/'source',spec['capture_limits'],spec)
        for name,wanted in built['runtime'].items():
            assert pin(BASE/'runtime'/name)==wanted and pin(Path(built['test_bin'])/name)==wanted,name


if __name__=='__main__':
    assert sys.platform=='linux' and not sys.flags.optimize
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
