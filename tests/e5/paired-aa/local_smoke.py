"""Single-use Windows functional smoke, with CPU affinity inherited before CLR startup."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,time,traceback
import psutil
from prepare_inputs import sha,pin,write_new,CORE,PROTOBUF

def main(base,root):
    assert os.name=='nt' and not (base/'smoke-process.json').exists()
    assert sha(base/'bin/Lokad.Onnx.dll')==CORE and sha(base/'bin/Google.Protobuf.dll')==PROTOBUF
    assert psutil.virtual_memory().available>=9*1024**3,'Insufficient memory before smoke'
    source=base/'smoke-source';source.mkdir()
    for name in ('Bridge.cs','Host.cs','Bridge.csproj','Host.csproj','local_smoke.py','prepare_inputs.py'):
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    record=dict(schema=1,started=time.time(),complete=False,
        source={p.name:pin(p) for p in source.iterdir()},
        binaries={p.name:pin(p) for p in (base/'bin').iterdir() if p.suffix in ('.dll','.json')},samples=[])
    parent=psutil.Process();parent.cpu_affinity([2])
    command=['dotnet',str(base/'bin/PairedHost.dll'),str(root/'models/multilingual-e5-small/model.onnx'),
        str(base/'inputs/e5-8tok.json'),str(base/'smoke'), '0','0','smoke']
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    child=None;birth=None
    try:
        with (base/'smoke.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            p=psutil.Process(child.pid);birth=p.create_time();record.update(pid=child.pid,birth=birth,command=command)
            parent.cpu_affinity([0]);start=time.monotonic()
            print('Started functional smoke',child.pid,birth,flush=True)
            while child.poll() is None:
                try:
                    rss=p.memory_info().rss;affinity=p.cpu_affinity();observed=p.create_time()
                except psutil.NoSuchProcess:
                    if child.poll() is not None:break
                    raise
                sample=dict(seconds=time.monotonic()-start,rss=rss,affinity=affinity,birth=observed,available=psutil.virtual_memory().available)
                record['samples'].append(sample)
                assert observed==birth and affinity==[2] and rss<8*1024**3 and sample['seconds']<300 and sample['available']>=1024**3,sample
                time.sleep(.5)
            record['code']=child.wait();assert record['code']==0
        assert not psutil.pid_exists(child.pid) or psutil.Process(child.pid).create_time()!=birth
        record['complete']=True
    except BaseException:
        record['error']=traceback.format_exc();raise
    finally:
        if child is not None and child.poll() is None:
            p=psutil.Process(child.pid)
            if p.create_time()==birth:p.kill()
            child.wait(timeout=10)
        record['ended']=time.time();write_new(base/'smoke-process.json',record)
    print('Functional smoke complete; both private engines agree and hold their inputs/outputs.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    a=p.parse_args();main(a.artifact.resolve(),a.root.resolve())
