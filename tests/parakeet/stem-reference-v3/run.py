"""Once-only local reference schedule; independent of both VM controllers."""
from common import *
import shutil
import subprocess
import time
import traceback


def main():
    spec=read(BASE/'manifest.json');verify(spec);assert not (BASE/'processes.json').exists()
    parent=psutil.Process();previous=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(complete=False,code=None,manifest=pin(BASE/'manifest.json'),supervisor=dict(pid=parent.pid,birth=parent.create_time()),runs=[])
    save(BASE/'processes.json',state)
    try:
        for job in JOBS:
            preflight=dict(available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
            assert preflight['available']>=LIMITS['preflight_available'] and preflight['disk']>=LIMITS['disk'],preflight
            folder=BASE/'process'/job['id'];folder.mkdir(parents=True,exist_ok=False)
            run=dict(job=job,complete=False,code=None,preflight=preflight,samples=0,peak_rss=0);state['runs'].append(run)
            child=None;identity=None;started=time.monotonic()
            try:
                env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
                command=[sys.executable,'-X','utf8','-B',str(Path(__file__).with_name('worker.py')),job['id']];run['command']=command
                with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
                    parent.cpu_affinity([2])
                    try:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                    finally:parent.cpu_affinity([0])
                    p=psutil.Process(child.pid);identity=dict(pid=p.pid,birth=p.create_time());run['worker']=identity;save(BASE/'processes.json',state)
                    while child.poll() is None:
                        try:
                            assert p.create_time()==identity['birth'] and not p.children(recursive=True)
                            sample=dict(seconds=time.monotonic()-started,pid=p.pid,birth=p.create_time(),rss=p.memory_info().rss,
                                        available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free,affinity=p.cpu_affinity())
                        except psutil.NoSuchProcess:continue
                        samples.write(json.dumps(sample)+'\n');samples.flush();run['samples']+=1;run['peak_rss']=max(run['peak_rss'],sample['rss'])
                        assert sample['seconds']<LIMITS['seconds'] and sample['rss']<LIMITS['rss']
                        assert sample['available']>=LIMITS['available'] and sample['disk']>=LIMITS['disk'] and sample['affinity']==[2]
                        time.sleep(.25)
                    run['code']=child.wait();assert run['code']==0,(job['id'],run['code'])
            except BaseException:
                run['error']=traceback.format_exc()
                if child is not None and child.poll() is None and identity is not None and not absent(identity):
                    process=psutil.Process(identity['pid'])
                    for member in process.children(recursive=True):member.kill()
                    process.kill();child.wait(timeout=10)
                raise
            finally:
                run.update(complete=True,seconds=time.monotonic()-started)
                if child is not None:run['code']=child.poll()
                save(BASE/'processes.json',state)
            assert absent(identity);print(json.dumps(dict(job=job['id'],seconds=run['seconds'],peak_rss=run['peak_rss'])),flush=True)
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(BASE/'processes.json',state);parent.cpu_affinity(previous)


if __name__=='__main__':main()
