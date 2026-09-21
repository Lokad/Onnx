"""Two sequential local workers; preserve every sample and failed attempt."""
from common import *
import shutil
import subprocess
import time
import traceback


def main():
    spec=read(BASE/'manifest.json');verify(spec);assert not (BASE/'processes.json').exists()
    parent=psutil.Process();previous=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(complete=False,code=None,supervisor=dict(pid=parent.pid,birth=parent.create_time()),runs=[])
    save(BASE/'processes.json',state)
    try:
        for role in JOBS:
            preflight=dict(available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
            assert preflight['available']>=LIMITS['preflight_available'] and preflight['disk']>=LIMITS['disk'],preflight
            folder=BASE/'process'/role;folder.mkdir(parents=True,exist_ok=False)
            row=dict(role=role,complete=False,code=None,preflight=preflight,samples=0,peak_rss=0)
            state['runs'].append(row);save(BASE/'processes.json',state)
            child=None;identity=None;started=time.monotonic()
            try:
                command=['dotnet',str(BASE/'runtimes'/role/'ProfilerAllocation.dll'),str(ROOT),str(BASE/'manifest.json'),role]
                row['command']=command
                with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
                    parent.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=ROOT,env=clean_env(),stdout=out,stderr=err,
                        creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                    finally:parent.cpu_affinity([0])
                    process=psutil.Process(child.pid);identity=dict(pid=child.pid,birth=process.create_time());row['worker']=identity
                    save(BASE/'processes.json',state)
                    while child.poll() is None:
                        try:
                            assert process.create_time()==identity['birth'] and not process.children(recursive=True)
                            s=dict(seconds=time.monotonic()-started,rss=process.memory_info().rss,available=psutil.virtual_memory().available,
                                disk=shutil.disk_usage(BASE).free,affinity=process.cpu_affinity(),**identity)
                        except psutil.NoSuchProcess:continue
                        samples.write(json.dumps(s)+'\n');samples.flush();row['samples']+=1;row['peak_rss']=max(row['peak_rss'],s['rss'])
                        assert s['seconds']<LIMITS['seconds'] and s['rss']<LIMITS['rss'] and s['available']>=LIMITS['available']
                        assert s['disk']>=LIMITS['disk'] and s['affinity']==[2]
                        time.sleep(.25)
                    row['code']=child.wait();assert row['code']==0,(role,row['code'])
            except BaseException:
                if child is not None and child.poll() is None and identity is not None and not absent(identity):
                    process=psutil.Process(identity['pid'])
                    for p in process.children(recursive=True):p.kill()
                    process.kill();child.wait(timeout=10)
                raise
            finally:
                row.update(complete=True,seconds=time.monotonic()-started)
                if child is not None:row['code']=child.poll()
                save(BASE/'processes.json',state)
            assert absent(identity);print(json.dumps(row),flush=True)
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'processes.json',state);parent.cpu_affinity(previous)


if __name__=='__main__':main()
