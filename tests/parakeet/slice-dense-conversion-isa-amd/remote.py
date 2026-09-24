"""Use DOTNET_EnableAVX512=0 and require the existing consumed-mode assertion."""
from pathlib import Path
import common
from common import BASE,DOTNET,pin,read,save,live,idle,verify,job


def capture(state,env,spec):
    assert not live(spec['original_owner']) and all(not live(p) for p in spec['original_members'])
    original=Path(spec['original']);built=read(original/'built.json')
    assert built['core']==spec['core'] and built['consumer']==spec['consumer']
    project=original/'source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
    environment=dict(env,DENSE_COPY_CORE_SHA=spec['core']['sha256'],DENSE_COPY_AVX512='0',
        DOTNET_EnableAVX512='0',DOTNET_CLI_HOME=str(original/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',
        DOTNET_CLI_TELEMETRY_OPTOUT='1',NUGET_PACKAGES=str(original/'packages'),NUGET_HTTP_CACHE_PATH=str(original/'http-cache'),
        MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0')
    assert 'DOTNET_EnableAVX512F' not in environment
    command=[DOTNET,'test',project,'-c','Release','--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false',
        '-p:NuGetAudit=false','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','--no-build','--no-restore',
        '--logger','trx;LogFileName=tensors-256.trx','--results-directory',BASE/'logs']
    state['instruction_environment']=spec['instruction_environment'];save(BASE/'capture-state.json',state)
    job(state,'tensors-256',command,environment,original/'source',spec['capture_limits'],spec)
    verify()


if __name__=='__main__':
    common.capture=capture
    raise SystemExit(common.main())
