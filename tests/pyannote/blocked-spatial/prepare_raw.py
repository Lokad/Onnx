"""Build and qualify the first isolated spatial prototype; never time it here."""
import importlib.util
import json
from pathlib import Path
import shutil
import traceback
from generate import generate

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-blocked-spatial-raw-20260922'
SELECTED=ROOT/'artifacts/pyannote-single-panel-models-20260922/runtime'
CENSUS=ROOT/'artifacts/pyannote-blocked-spatial-census-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
FEED=ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
spec=importlib.util.spec_from_file_location('blocked_raw_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor);monitor.BASE=BASE
pin,read,save,verify,psutil=monitor.pin,monitor.read,monitor.save,monitor.verify,monitor.psutil


def rel(p):return p.relative_to(ROOT).as_posix()


def main():
    assert not BASE.exists()
    assert pin(CENSUS/'closed.json')['sha256']=='1fe735dde2847feee9af63037f862700c30f961fe5d056519e4c15a74a4432d1'
    census=read(CENSUS/'closed.json');assert census['passed'];verify(census['files'])
    assert pin(SELECTED/'Lokad.Onnx.dll')['sha256']=='1279b4b662241db2404fa4875eae20eaa924f15677a85655f99b8f81cd24b309'
    BASE.mkdir();(BASE/'logs').mkdir();(BASE/'output').mkdir();(BASE/'tools').mkdir()
    source=BASE/'source';source.mkdir()
    for p in TOOLS.iterdir():
        if p.is_file():shutil.copy2(p,BASE/'tools'/p.name)
    for name in ['BlockedSpatial','Probe']:shutil.copy2(TOOLS/(name+'.cs.txt'),source/(name+'.cs'))
    generate(source/'GeneratedKernels.cs')
    runtime=BASE/'runtime';runtime.mkdir()
    for name in ['Lokad.Onnx.dll','Google.Protobuf.dll']:shutil.copy2(SELECTED/name,runtime/name)
    project=source/'BlockedSpatialProbe.csproj'
    project.write_text('''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><AllowUnsafeBlocks>true</AllowUnsafeBlocks><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><NuGetAudit>false</NuGetAudit></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../runtime/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Google.Protobuf"><HintPath>../runtime/Google.Protobuf.dll</HintPath></Reference></ItemGroup></Project>''',encoding='utf8')
    plan=ROOT/'.agent/m13-pyannote-blocked-spatial-20260922.md';shutil.copy2(plan,BASE/'prospective-plan.md')
    files={rel(p):pin(p) for p in BASE.rglob('*') if p.is_file()}
    files.update({rel(p):pin(p) for p in [MONITOR,CENSUS/'closed.json']})
    save(BASE/'inputs.json',dict(files=files,scope='Raw/layout/caller correctness only; no performance measurement',seed=941706))
    own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    path=BASE/'preparation.json';save(path,state);flags=monitor.FLAGS+['-p:NuGetAudit=false']
    try:
        monitor.worker(state,path,'restore',['dotnet','restore',project,*flags,'--source',FEED,'--packages',BASE/'packages'],ROOT,[0],8,8,900,True,source)
        monitor.worker(state,path,'build',['dotnet','build',project,'-c','Release',*flags,'--no-restore','--disable-build-servers'],ROOT,[0],8,8,900,True,source)
        built=source/'bin/Release/net10.0'
        assert pin(built/'Lokad.Onnx.dll')==pin(runtime/'Lokad.Onnx.dll')
        monitor.worker(state,path,'qualify-256',['dotnet',built/'BlockedSpatialProbe.dll','256',BASE/'output/256.json'],ROOT,[0,1],12,8,900,False,BASE/'output')
        result=read(BASE/'output/256.json')
        assert result['geometries']==312 and result['cases']==2648 and result['rejected']==10
        assert result['passed']==(result['failed_cases']==0)==(state['runs'][-1]['code']==0)
        verify(files)
        save(BASE/'verified.json',dict(complete=True,passed=result['passed'],files=files,report=pin(BASE/'output/256.json'),
            core=pin(runtime/'Lokad.Onnx.dll'),consumer=pin(built/'BlockedSpatialProbe.dll'),no_performance_measurement=True))
        state['code']=0 if result['passed'] else 1
        print(json.dumps({k:result[k] for k in ['passed','cases','failed_cases','scalar_differences','production_differences']}))
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(path,state)
    return state['code']


if __name__=='__main__':raise SystemExit(main())
