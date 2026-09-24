"""Build one consumer, then measure existing copy paths in four fresh processes."""
import importlib.util
from pathlib import Path
import shutil
import common
from common import BASE, DOTNET, pin, read, save, job, live, idle, verify


def build(state,env,spec):
    for name in ['tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    env=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
        NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',
        DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))
    flags=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false',
        '-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    source=BASE/'source';project=source/'CopyCost.csproj';limits=spec['build_limits']
    job(state,'sdk-version',[DOTNET,'--version'],env,source,limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    job(state,'restore',[DOTNET,'restore',project,*flags,'--source',spec['feed']],env,source,limits,spec)
    job(state,'build',[DOTNET,'build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers','-o',BASE/'runtime'],env,source,limits,spec)
    assert pin(BASE/'runtime/Lokad.Onnx.dll')==spec['core']
    manifest=read(BASE/'cases.json')
    manifest.update(consumer_sha256=pin(BASE/'runtime/CopyCost.dll')['sha256'])
    save(BASE/'manifest.json',manifest)
    save(BASE/'built.json',dict(core=spec['core'],consumer=pin(BASE/'runtime/CopyCost.dll'),manifest=pin(BASE/'manifest.json'),
        runtime={p.relative_to(BASE).as_posix():pin(p) for p in (BASE/'runtime').iterdir() if p.is_file()}))


def capture(state,env,spec):
    review=read(BASE/'build-review.json');built=read(BASE/'built.json')
    assert review['passed'] and review['built']==pin(BASE/'built.json')
    assert pin(BASE/'manifest.json')==built['manifest']
    for name,wanted in built['runtime'].items():assert pin(BASE/name)==wanted,name
    loader=importlib.util.spec_from_file_location('cpu_accounting',spec['accounting'])
    accounting=importlib.util.module_from_spec(loader);loader.loader.exec_module(accounting)
    for index,mode in enumerate(spec['order']):
        verify();name=f'{index:02}-{mode}';output=BASE/'results'/name
        before=accounting.snapshot()
        job(state,name,[DOTNET,BASE/'runtime/CopyCost.dll',BASE/'manifest.json',BASE/'constant.bin',output,mode],
            env,BASE,spec['capture_limits'],spec,output)
        after=accounting.snapshot();row=state['runs'][-1]
        row.update(cpu_before=before,cpu_after=after,accounting=accounting.foreign_fraction(before,after,state['supervisor']['pid']))
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction']<=.01
        save(BASE/'capture-state.json',state)
        result=read(output/'result.json');assert result['passed'] and result['records']==1920


if __name__=='__main__':
    common.build=build;common.capture=capture
    raise SystemExit(common.main())
