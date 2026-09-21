"""Bound one owned capture or probe worker; no automatic inference retry."""
import os
import shutil
import subprocess
import traceback
from common import *


def main():
    assert len(sys.argv) in (2,3) and sys.argv[1] in ('capture','probe');mode=sys.argv[1]
    ordinal=int(sys.argv[2]) if mode=='probe' else None
    if mode=='probe':assert ordinal in range(4)
    name=mode if ordinal is None else f'probe-{ordinal}'
    prepared=read(BASE/'prepared.json');assert prepared['passed'];verify(prepared['files'])
    if mode=='probe':
        prior=read(BASE/'capture-closed.json');assert prior['passed'];verify(prior['files'])
        if ordinal>0:
            previous=read(BASE/f'probe-{ordinal-1}-state.json');assert previous['complete'] and previous['passed']
            for key in ('supervisor','worker'):terminal(previous[key])
    state_path=BASE/(name+'-state.json');assert not state_path.exists()
    own=psutil.Process();affinity=own.cpu_affinity();own.cpu_affinity([0]);child=None
    state=dict(complete=False,code=None,mode=mode,ordinal=ordinal,supervisor=dict(pid=own.pid,birth=own.create_time()),samples=0,peak_rss=0,preflight_observations=[])
    minimum=(14 if mode=='capture' else 4)*1024**3;limit=(12 if mode=='capture' else 2)*1024**3
    try:
        start=time.monotonic()
        while True:
            row=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
            state['preflight_observations'].append(row);save(state_path,state)
            assert row['seconds']<900 and row['disk']>=20*1024**3
            if row['available']>=minimum:break
            time.sleep(15)
        state['preflight']=row
        env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        command=['dotnet',str(BASE/'bin/Probe.dll'),mode,str(ROOT),str(BASE/(mode+'-manifest.json')),str(BASE/name)]
        if ordinal is not None:command.append(str(ordinal))
        with (BASE/(name+'-stdout.txt')).open('x') as out,(BASE/(name+'-stderr.txt')).open('x') as err,(BASE/(name+'-resources.jsonl')).open('x') as resources:
            own.cpu_affinity([2])
            try:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
            finally:own.cpu_affinity([0])
            worker=psutil.Process(child.pid);state['worker']=dict(pid=worker.pid,birth=worker.create_time());start=time.monotonic()
            while child.poll() is None:
                try:
                    assert worker.create_time()==state['worker']['birth'] and not worker.children(recursive=True)
                    row=dict(seconds=time.monotonic()-start,rss=worker.memory_info().rss,available=psutil.virtual_memory().available,
                        disk=shutil.disk_usage(BASE).free,affinity=worker.cpu_affinity(),bytes=sum(p.stat().st_size for p in (BASE/name).rglob('*') if p.is_file()))
                except psutil.NoSuchProcess:
                    if child.poll() is not None:break
                    raise
                resources.write(json.dumps(row)+'\n');resources.flush();state['samples']+=1;state['peak_rss']=max(state['peak_rss'],row['rss']);save(state_path,state)
                assert row['seconds']<600 and row['rss']<limit and row['available']>=1024**3 and row['disk']>=20*1024**3 and row['affinity']==[2]
                assert row['bytes']<=(512 if mode=='capture' else 64)*1024**2
                time.sleep(.25)
            state['code']=child.wait();assert state['code']==0,state['code']
        verify(prepared['files']);assert read(BASE/name/'result.json')['passed'];state['passed']=True
    except BaseException:
        state.update(code=1,error=traceback.format_exc())
        if child is not None and child.poll() is None:
            worker=psutil.Process(child.pid)
            if worker.create_time()==state['worker']['birth']:worker.kill();child.wait(timeout=15)
        raise
    finally:
        state['complete']=True;save(state_path,state);own.cpu_affinity(affinity)
    print(json.dumps(state))


if __name__=='__main__':main()
