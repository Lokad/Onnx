"""Reuse the checked consumer with detached startup and explicit child diagnostics."""
import os
import shutil
import subprocess
import time
import traceback
from qualify import ROOT,MANIFEST,pin,read,verify,save,terminal,psutil

PRIOR=ROOT/'artifacts/parakeet-packing-qualification-20260921'
BASE=ROOT/'artifacts/parakeet-packing-qualification-v2-20260921'


def main():
    original=read(PRIOR/'prepared.json');assert original['passed'];verify(original['files'])
    failure=read(PRIOR/'processes.json');assert failure['complete'] and failure['code']==1
    terminal(failure['supervisor'])
    for row in failure['runs']:terminal(row['worker'])
    assert not (PRIOR/'trace-output').exists()
    BASE.mkdir();(BASE/'logs').mkdir();shutil.copytree(PRIOR/'bin',BASE/'bin')
    files=dict(original['files'])
    for folder in (BASE/'bin',PRIOR/'logs'):
        for p in folder.iterdir():
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in (PRIOR/'prepared.json',PRIOR/'processes.json',__import__('pathlib').Path(__file__)):
        files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(original,files=files,failed_predecessor=pin(PRIOR/'processes.json')))
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    own=psutil.Process();previous=own.cpu_affinity();own.cpu_affinity([0])
    controller=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    try:
        for mode in ('trace','public'):
            child=None;state=dict(mode=mode,complete=False,code=None,samples=0,peak_rss=0,preflight_observations=[],members={})
            controller['runs'].append(state);beginning=time.monotonic();output=BASE/(mode+'-output')
            try:
                while True:
                    row=dict(seconds=time.monotonic()-beginning,available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
                    state['preflight_observations'].append(row);save(BASE/'processes.json',controller)
                    assert row['seconds']<900 and row['disk']>=20*1024**3
                    if row['available']>=14*1024**3:break
                    time.sleep(15)
                state['preflight']=row
                command=['dotnet',str(BASE/'bin/Profile.dll'),str(ROOT),str(MANIFEST),str(output),mode]
                with (BASE/'logs'/(mode+'.log')).open('x') as log,(BASE/'logs'/(mode+'.samples.jsonl')).open('x') as samples:
                    own.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                    finally:own.cpu_affinity([0])
                    worker=psutil.Process(child.pid);state['worker']=dict(pid=worker.pid,birth=worker.create_time());start=time.monotonic()
                    state['members'][str(worker.pid)]=state['worker']['birth']
                    while child.poll() is None:
                        try:
                            current_birth=worker.create_time();children=[]
                            for p in worker.children(recursive=True):
                                try:
                                    birth=p.create_time();state['members'][str(p.pid)]=birth
                                    children.append(dict(pid=p.pid,birth=birth,name=p.name(),command=p.cmdline()))
                                except psutil.NoSuchProcess:pass
                            row=dict(seconds=time.monotonic()-start,rss=worker.memory_info().rss,available=psutil.virtual_memory().available,
                                disk=shutil.disk_usage(BASE).free,affinity=worker.cpu_affinity(),children=children,current_birth=current_birth)
                        except psutil.NoSuchProcess:
                            if child.poll() is not None:break
                            raise
                        samples.write(__import__('json').dumps(row)+'\n');samples.flush();state['samples']+=1;state['peak_rss']=max(state['peak_rss'],row['rss']);save(BASE/'processes.json',controller)
                        assert current_birth==state['worker']['birth'] and not children,row
                        assert row['seconds']<1200 and row['rss']<12*1024**3 and row['available']>=1024**3 and row['disk']>=20*1024**3 and row['affinity']==[2]
                        time.sleep(.25)
                    state['code']=child.wait();assert state['code']==0,state['code']
                assert read(output/'result.json')['passed'];verify(files);state['passed']=True
            except BaseException:
                state['error']=traceback.format_exc()
                for pid,birth in reversed(list(state['members'].items())):
                    try:
                        p=psutil.Process(int(pid))
                        if p.create_time()==birth:p.kill()
                    except psutil.NoSuchProcess:pass
                if child is not None:child.wait(timeout=15)
                raise
            finally:
                state['complete']=True
                if child is not None:state['code']=child.poll()
                save(BASE/'processes.json',controller)
            print(mode,'completed',flush=True)
        controller['code']=0
    except BaseException:controller.update(code=1,error=traceback.format_exc());raise
    finally:controller['complete']=True;save(BASE/'processes.json',controller);own.cpu_affinity(previous)


if __name__=='__main__':main()
