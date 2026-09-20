"""Freeze and run two finite decoder-only metadata diagnostics with saved stage snapshots."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/whisper/memory-contracts'))
from prepare import pin,read,write
sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

BASE=ROOT/'artifacts/whisper-weight-metadata-20260920'
PRODUCT=ROOT/'artifacts/whisper-weight-sharing-20260920'
LIMITS=dict(seconds=120,rss=8*1024**3,available=4*1024**3,preflight=8*1024**3,disk=32*1024**2,preflight_disk=64*1024**2)


def main():
    assert psutil.__version__=='7.0.0' and os.name=='nt'
    failed=read(PRODUCT/'failure-closed.json');assert failed['closure_passed'] and not failed['campaign_passed']
    for name,wanted in failed['files'].items():assert pin(ROOT/name)==wanted,name
    BASE.mkdir();source=BASE/'source';source.mkdir();product=BASE/'product-bin';product.mkdir()
    for path in (PRODUCT/'product-bin').iterdir():shutil.copyfile(path,product/path.name)
    template=Path(__file__).with_name('Program.cs');inspector=PRODUCT/'inspection-v2/source/Program.cs'
    methods=inspector.read_text(encoding='utf-8').split('static object Snapshot(',1)[1]
    methods=methods.replace('n.Inputs, n.Outputs','Inputs=n.Inputs.ToArray(), Outputs=n.Outputs.ToArray()')
    (source/'Program.cs').write_text(template.read_text(encoding='utf-8')+'\nstatic object Snapshot('+methods,encoding='utf-8')
    project='''<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings></PropertyGroup><ItemGroup><Reference Include="Lokad.Onnx"><HintPath>../product-bin/Lokad.Onnx.dll</HintPath></Reference><Reference Include="Lokad.Onnx.Data"><HintPath>../product-bin/Lokad.Onnx.Data.dll</HintPath></Reference></ItemGroup></Project>'''
    (source/'WhisperWeightMetadata.csproj').write_text(project,encoding='utf-8')
    with (BASE/'build.log').open('x') as log:r=subprocess.run(['dotnet','build',str(source/'WhisperWeightMetadata.csproj'),'--tl:off','--nologo','-v','minimal','-c','Release','-o',str(BASE/'bin')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,'Diagnostic build'
    for path in product.iterdir():
        target=BASE/'bin'/path.name
        if target.exists():assert pin(target)==pin(path)
        else:shutil.copyfile(path,target)
    runtime=Path('C:/Program Files/dotnet/shared/Microsoft.NETCore.App/10.0.12');dotnet=Path(shutil.which('dotnet'))
    external={str(p):pin(p) for p in sorted(runtime.rglob('*')) if p.is_file()};external[str(dotnet)]=pin(dotnet)
    models=ROOT/'models/whisper-large-v3-turbo/onnx';census=read(PRODUCT/'weight-census.json')
    for name,wanted in census['models'].items():assert pin(models/name)==wanted;external[str(models/name)]=wanted
    for name in ['Program.cs','run.py']:shutil.copyfile(Path(__file__).with_name(name),BASE/name)
    shutil.copyfile(ROOT/'.agent/m4-whisper-weight-metadata-20260920.md',BASE/'prospective-plan.md')
    files={p.relative_to(BASE).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    write(BASE/'frozen.json',dict(files=files,external=external,limits=LIMITS,prior_failure=pin(PRODUCT/'failure-closed.json'),
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    parent=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=parent.pid,birth=parent.create_time()),runs=[],frozen=pin(BASE/'frozen.json'))
    def save():
        p=BASE/'identity.tmp';p.write_text(json.dumps(state,indent=2));p.replace(BASE/'identity.json')
    save()
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    try:
        for mode in ['unshared','shared']:
            folder=BASE/mode;folder.mkdir();available=psutil.virtual_memory().available;disk=psutil.disk_usage(str(BASE)).free
            assert available>=LIMITS['preflight'] and disk>=LIMITS['preflight_disk']
            row=dict(mode=mode,complete=False,code=None,started=time.time(),preflight_available=available,preflight_disk=disk,samples=0,peak_rss=0)
            state['runs'].append(row);save();child=None;start=time.monotonic();prior=parent.cpu_affinity()
            try:
                with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
                    parent.cpu_affinity([2])
                    try:child=subprocess.Popen([str(dotnet),str(BASE/'bin/WhisperWeightMetadata.dll'),str(models),mode,str(folder/'worker')],cwd=ROOT,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                    finally:parent.cpu_affinity(prior)
                    process=psutil.Process(child.pid);row['child']=dict(pid=child.pid,birth=process.create_time());save()
                    while child.poll() is None:
                        try:
                            assert process.create_time()==row['child']['birth']
                            sample=dict(seconds=time.monotonic()-start,rss=process.memory_info().rss,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(BASE)).free,affinity=process.cpu_affinity())
                        except psutil.NoSuchProcess:break
                        samples.write(json.dumps(sample)+'\n');samples.flush();row['samples']+=1;row['peak_rss']=max(row['peak_rss'],sample['rss']);save()
                        assert sample['seconds']<LIMITS['seconds'] and sample['rss']<LIMITS['rss'] and sample['available']>=LIMITS['available'] and sample['disk']>=LIMITS['disk'] and sample['affinity']==[2]
                        time.sleep(.5)
                    row['code']=child.wait();assert row['code']==0
                value=read(folder/'worker/result.json');assert value['passed'] and value['runtime']=='.NET 10.0.12' and value['affinity']==4 and value['processor_count']==1 and value['flags']=={}
                row['result']=pin(folder/'worker/result.json')
            except BaseException as error:
                row['error']=repr(error)
                if child is not None and child.poll() is None:
                    if psutil.Process(child.pid).create_time()==row['child']['birth']:child.kill();child.wait(timeout=10)
                raise
            finally:row.update(complete=True,seconds=time.monotonic()-start,ended=time.time());save()
        for name,wanted in files.items():assert pin(BASE/name)==wanted,name
        for name,wanted in external.items():assert pin(Path(name))==wanted,name
        state['code']=0
    except BaseException as error:state['error']=repr(error);traceback.print_exc();raise
    finally:state['complete']=True;save()
    print(json.dumps(dict(passed=True,workers=state['runs'])))


if __name__=='__main__':main()
