"""Reuse the tensor build; keep measured Data and verify actual instruction modes."""
from pathlib import Path
import shutil
import common
import candidate_build
from common import BASE, DOTNET, pin, read, save, live, idle, verify, job


def build(state,env,spec):
    candidate_build.build(state,env,spec)
    runtime=BASE/'source/runtime-observed'
    assert not (runtime/'Lokad.Onnx.Data.dll').exists()
    shutil.copy2(Path(spec['prior'])/'Lokad.Onnx.Data.dll',runtime/'Lokad.Onnx.Data.dll')
    assert pin(runtime/'Lokad.Onnx.Data.dll')==spec['data']
    built=read(BASE/'built.json');built['data']=spec['data']
    built['runtime_files']['source/runtime-observed/Lokad.Onnx.Data.dll']=spec['data']
    save(BASE/'built.json',built)


def capture(state,env,spec):
    review=read(BASE/'build-review.json');assert review['passed'] and review['built']==pin(BASE/'built.json')
    built=read(BASE/'built.json');project=BASE/'source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
    for mode in ['512','256']:
        for name,wanted in built['runtime_files'].items():assert pin(BASE/name)==wanted
        environment=dict(env,SLICE_CORE_SHA=built['core']['sha256'],SLICE_AVX512='1' if mode=='512' else '0',
            DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
            NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),
            MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0')
        if mode=='256':environment['DOTNET_EnableAVX512']='0'
        command=[DOTNET,'test',project,'-c','Release',*candidate_build.FLAGS,'--no-build','--no-restore',
            '-p:OutputPath='+str(BASE/'source/runtime-observed')+'/', '-p:AppendTargetFrameworkToOutputPath=false',
            '--logger','trx;LogFileName=tensors-'+mode+'.trx','--results-directory',BASE/'logs']
        job(state,'tensors-'+mode,command,environment,BASE/'source',spec['capture_limits'],spec)
        for name,wanted in built['runtime_files'].items():assert pin(BASE/name)==wanted


if __name__=='__main__':
    common.build,common.capture=build,capture
    raise SystemExit(common.main())
