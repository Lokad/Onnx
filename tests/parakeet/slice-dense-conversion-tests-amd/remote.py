"""Compile corrected tests against the exact inspected Core, then run both modes."""
import shutil
import common
from common import BASE,DOTNET,pin,read,save,live,idle,verify,job

FLAGS=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
       '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']


def environment(env):
    return dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
        NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))


def build(state,env,spec):
    assert not live(spec['original_owner']) and all(not live(p) for p in spec['original_members'])
    for name in ['tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    env=environment(env);limits=spec['build_limits']
    project=BASE/'source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
    job(state,'tensors-restore',[DOTNET,'restore',project,*FLAGS,'--source','/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed','--packages',BASE/'packages'],env,project.parent,limits,spec)
    job(state,'tensors-build',[DOTNET,'build',project,'-c','Release',*FLAGS,'--no-restore','--disable-build-servers'],env,project.parent,limits,spec)
    original=project.parent/'bin/Release/net10.0';runtime=BASE/'runtime';runtime.mkdir()
    for path in original.iterdir():
        if path.is_file():shutil.copy2(path,runtime/path.name)
    assert pin(runtime/'Lokad.Onnx.dll')==spec['core']
    assert not (BASE/'source/src/Lokad.Onnx/bin').exists()
    save(BASE/'built.json',dict(core=pin(runtime/'Lokad.Onnx.dll'),consumer=pin(runtime/'Lokad.Onnx.Tensors.Tests.dll'),
        runtime_files={p.relative_to(BASE).as_posix():pin(p) for p in runtime.iterdir()},test_bin=str(original),product_rebuilt=False))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');assert review['passed'] and review['built']==pin(BASE/'built.json')
    built=read(BASE/'built.json');project=BASE/'source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
    for mode in ['512','256']:
        verify()
        for name,wanted in built['runtime_files'].items():
            assert pin(BASE/name)==wanted
            assert pin(project.parent/'bin/Release/net10.0'/name.split('/')[-1])==wanted
        env_mode=dict(environment(env),DENSE_COPY_CORE_SHA=built['core']['sha256'],DENSE_COPY_AVX512='1' if mode=='512' else '0',DOTNET_EnableAVX512F='1' if mode=='512' else '0')
        command=[DOTNET,'test',project,'-c','Release',*FLAGS,'--no-build','--no-restore',
            '--logger','trx;LogFileName=tensors-'+mode+'.trx','--results-directory',BASE/'logs']
        job(state,'tensors-'+mode,command,env_mode,BASE/'source',spec['capture_limits'],spec)
        verify()
        for name,wanted in built['runtime_files'].items():assert pin(BASE/name)==wanted


if __name__=='__main__':
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
