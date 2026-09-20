"""Bounded local, preparation-only proof of actual weight storage and unchanged prepared graphs."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time,traceback
from prepare import ROOT,BASE,pin,write
sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def main():
    assert psutil.__version__=='7.0.0'
    built=json.loads((BASE/'built.json').read_text());assert built['tests_passed']
    for name,wanted in built['files'].items():assert pin(BASE/name)==wanted,name
    folder=BASE/'inspection-v2';assert not folder.exists();folder.mkdir();source=folder/'source';source.mkdir()
    shutil.copyfile(Path(__file__).with_name('Inspect.cs'),source/'Program.cs')
    project='''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../../product-bin/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Lokad.Onnx.Data"><HintPath>../../product-bin/Lokad.Onnx.Data.dll</HintPath></Reference></ItemGroup></Project>'''
    (source/'WhisperWeightInspect.csproj').write_text(project,encoding='utf-8')
    with (folder/'build.log').open('x') as log:r=subprocess.run(['dotnet','build',str(source/'WhisperWeightInspect.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(folder/'bin')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,'Inspection consumer build'
    for p in (BASE/'product-bin').iterdir():
        target=folder/'bin'/p.name
        if target.exists():assert pin(target)==pin(p)
        else:shutil.copyfile(p,target)
    models=ROOT/'models/whisper-large-v3-turbo/onnx';census=json.loads((BASE/'weight-census.json').read_text())
    for name,wanted in census['models'].items():assert pin(models/name)==wanted,name
    limits=dict(seconds=120,rss=8*1024**3,available=4*1024**3,preflight=8*1024**3)
    assert psutil.virtual_memory().available>=limits['preflight']
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    state=dict(complete=False,code=None,started=time.time(),limits=limits,preflight_available=psutil.virtual_memory().available,
        files={p.relative_to(folder).as_posix():pin(p) for p in sorted(folder.rglob('*')) if p.is_file()},models=census['models'],samples=0,peak_rss=0)
    def save():
        target=folder/'identity.tmp';target.write_text(json.dumps(state,indent=2));target.replace(folder/'identity.json')
    process=None;start=time.monotonic();save()
    try:
        with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
            process=subprocess.Popen(['dotnet',str(folder/'bin/WhisperWeightInspect.dll'),str(models),str(folder/'result.json')],env=env,cwd=ROOT,
                stdout=out,stderr=err,stdin=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            child=psutil.Process(process.pid);state['child']=dict(pid=process.pid,birth=child.create_time());child.cpu_affinity([2]);save()
            process.stdin.write('GO\n');process.stdin.flush();process.stdin.close()
            while process.poll() is None:
                assert child.create_time()==state['child']['birth']
                try:
                    row=dict(seconds=time.monotonic()-start,rss=child.memory_info().rss,available=psutil.virtual_memory().available,affinity=child.cpu_affinity())
                except psutil.NoSuchProcess:break
                samples.write(json.dumps(row)+'\n');samples.flush();state['samples']+=1;state['peak_rss']=max(state['peak_rss'],row['rss']);save()
                assert row['seconds']<limits['seconds'] and row['rss']<limits['rss'] and row['available']>=limits['available'] and row['affinity']==[2]
                time.sleep(.5)
            state['code']=process.wait(timeout=10);assert state['code']==0
        value=json.loads((folder/'result.json').read_text());assert value['passed'] and not value['inference'] and value['flags']=={}
        for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','WhisperWeightInspect.dll')]:assert value[key]==pin(folder/'bin'/name)['sha256']
        for name,wanted in state['files'].items():assert pin(folder/name)==wanted,name
        for name,wanted in census['models'].items():assert pin(models/name)==wanted,name
        state['result']=pin(folder/'result.json')
    except BaseException as error:
        state['error']=repr(error)
        if process is not None and process.poll() is None:
            if psutil.Process(process.pid).create_time()==state['child']['birth']:process.kill();process.wait(timeout=10)
        traceback.print_exc();raise
    finally:
        state.update(complete=True,ended=time.time(),seconds=time.monotonic()-start);save()
    print(json.dumps(dict(passed=True,inference=False,shared=value['logical_shared_bytes'],unique_before=value['before']['unique_payload_bytes'],unique_after=value['after']['unique_payload_bytes'],peak_rss=state['peak_rss'],result=state['result'])))


if __name__=='__main__':main()
