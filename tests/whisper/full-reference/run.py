"""One fresh, identified reference worker at a time; two-step gated campaign."""
import argparse,os,shutil,subprocess,time,traceback
from common import *
packages()
import psutil

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--phase',choices=['bridge','remaining'],required=True)
    args=parser.parse_args();base=Path(args.artifact).resolve();spec=read(base/'manifest.json');verify(spec['files'])
    for path,wanted in spec['numerical_files'].items():assert pin(path)==wanted,path
    assert spec['limits']==LIMITS and spec['thread_environment']==THREADS and len(spec['jobs'])==84
    if args.phase=='remaining':
        gate=read(base/'bridge-gate.json');assert gate['passed'] and gate['manifest']==pin(base/'manifest.json')
        for name,wanted in gate['files'].items():assert pin(base/name)==wanted,name
        previous=read(base/'bridge.json');assert absent(previous['supervisor']) and all(absent(r['worker']) for r in previous['runs'])
    jobs=spec['jobs'][:2] if args.phase=='bridge' else spec['jobs'][2:]
    path=base/(args.phase+'.json');assert not path.exists();parent=psutil.Process();old_affinity=parent.cpu_affinity()
    state=dict(phase=args.phase,manifest=pin(base/'manifest.json'),supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,code=None,runs=[])
    save(path,state);parent.cpu_affinity([0])
    try:
        for job in jobs:
            available=psutil.virtual_memory().available;free=shutil.disk_usage(base).free
            assert available>=LIMITS['preflight_available'] and free>=LIMITS['disk'],('Preflight',job['id'],available,free)
            process_folder=base/'process'/job['id'];process_folder.mkdir(parents=True);child=None;identity=None;start=time.monotonic()
            record=dict(job=job,started=time.time(),preflight_available=available,preflight_disk=free,complete=False,code=None,samples=0)
            state['runs'].append(record);save(path,state)
            try:
                env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}|THREADS
                command=[sys.executable,'-X','utf8','-B',str(Path(__file__).with_name('worker.py')),'--artifact',str(base),'--job',job['id']]
                with (process_folder/'stdout.txt').open('x') as stdout,(process_folder/'stderr.txt').open('x') as stderr,(process_folder/'samples.jsonl').open('x') as samples:
                    parent.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.DETACHED_PROCESS)
                    finally:parent.cpu_affinity([0])
                    process=psutil.Process(child.pid);identity=dict(pid=process.pid,birth=process.create_time());record['worker']=identity;record['command']=command;save(path,state)
                    while child.poll() is None:
                        try:
                            assert process.create_time()==identity['birth'];assert not process.children(recursive=True),'Unexpected child'
                            row=dict(seconds=time.monotonic()-start,pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,
                                available=psutil.virtual_memory().available,affinity=process.cpu_affinity())
                        except psutil.NoSuchProcess:continue
                        samples.write(json.dumps(row)+'\n');samples.flush();record['samples']+=1
                        assert row['seconds']<LIMITS['seconds'] and row['rss']<LIMITS['rss'] and row['available']>=LIMITS['available'] and row['affinity']==[2],row
                        time.sleep(.25)
                    record['code']=child.returncode;assert child.returncode==0,('Worker exit',job['id'],child.returncode)
            except BaseException:
                record['error']=traceback.format_exc()
                if child is not None and child.poll() is None:
                    p=psutil.Process(child.pid)
                    if p.create_time()==identity['birth']:
                        for descendant in p.children(recursive=True):descendant.kill()
                        p.kill()
                    child.wait(timeout=10)
                raise
            finally:
                record['ended']=time.time();record['seconds']=time.monotonic()-start;record['complete']=True
                if child is not None:record['code']=child.poll()
                save(path,state)
            print(json.dumps(dict(completed=len(state['runs']),total=len(jobs),job=job['id'],seconds=record['seconds'])),flush=True)
        state['code']=0
    except BaseException:
        state['error']=traceback.format_exc();state['code']=1;raise
    finally:
        parent.cpu_affinity(old_affinity);state['complete']=True;state['ended']=time.time();save(path,state)

if __name__=='__main__':main()
