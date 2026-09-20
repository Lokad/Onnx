"""Run a single CPU2 worker, with an identified CPU0 resource supervisor."""
import argparse,os,shutil,subprocess,time,traceback
from common import *

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);args=parser.parse_args()
    base=Path(args.artifact).resolve();spec=read(base/'manifest.json');assert spec['limits']==LIMITS
    assert not (base/'campaign.json').exists();verify(spec['files']);ps=psutil_module();parent=ps.Process();original_affinity=parent.cpu_affinity()
    pre=dict(available=ps.virtual_memory().available,disk=shutil.disk_usage(base).free)
    assert pre['available']>=LIMITS['preflight_available'] and pre['disk']>=LIMITS['disk'],pre
    state=dict(complete=False,code=None,supervisor=dict(pid=parent.pid,birth=parent.create_time()),preflight=pre,manifest=pin(base/'manifest.json'),started=time.time())
    save(base/'campaign.json',state);child=None;worker=None;start=time.monotonic()
    try:
        env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}|THREAD_ENV
        command=[sys.executable,'-X','utf8','-B',str(Path(__file__).with_name('worker.py')),'--artifact',str(base)]
        with (base/'stdout.txt').open('x') as stdout,(base/'stderr.txt').open('x') as stderr,(base/'samples.jsonl').open('x') as samples:
            parent.cpu_affinity([2])
            try:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.DETACHED_PROCESS)
            finally:parent.cpu_affinity([0])
            worker=ps.Process(child.pid);identity=dict(pid=worker.pid,birth=worker.create_time());state['worker']=identity;state['command']=command;save(base/'campaign.json',state)
            while child.poll() is None:
                try:
                    assert worker.create_time()==identity['birth'];descendants=worker.children(recursive=True);assert not descendants
                    row=dict(seconds=time.monotonic()-start,pid=worker.pid,birth=worker.create_time(),rss=worker.memory_info().rss,
                        available=ps.virtual_memory().available,affinity=worker.cpu_affinity(),threads=worker.num_threads())
                except ps.NoSuchProcess:continue
                samples.write(json.dumps(row)+'\n');samples.flush()
                assert row['seconds']<LIMITS['seconds'] and row['rss']<LIMITS['rss'] and row['available']>=LIMITS['available'] and row['affinity']==[2],row
                time.sleep(.25)
            state['code']=child.returncode;assert child.returncode==0,child.returncode
    except BaseException:
        state['error']=traceback.format_exc()
        if child is not None and child.poll() is None:
            if worker is not None and worker.create_time()==state['worker']['birth']:
                for p in worker.children(recursive=True):p.kill()
                worker.kill()
            child.wait(timeout=10)
        raise
    finally:
        parent.cpu_affinity(original_affinity);state['complete']=True;state['ended']=time.time();state['seconds']=time.monotonic()-start
        if child is not None:state['code']=child.poll()
        save(base/'campaign.json',state)
    print(json.dumps(dict(complete=True,code=state['code'],seconds=state['seconds'],worker=state['worker'])))

if __name__=='__main__':main()
