"""Run the fixed fingerprint proof or four-visit timing schedule with owned process guards."""
from pathlib import Path
import argparse,json,os,subprocess,time,traceback
import psutil
from generate import pin

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)

def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--mode',choices=['proof','timing'],required=True)
    p.add_argument('--label',default='');a=p.parse_args()
    assert not a.label or (a.mode=='proof' and a.label.replace('-','').isalnum())
    base=a.artifact.resolve();model=a.model.resolve();generation=json.loads((base/'generation.json').read_text())
    assert pin(model)==generation['model']
    for name,wanted in generation['files'].items():assert pin(base/name)==wanted,name
    binaries={p.name:pin(p) for p in (base/'bin').iterdir() if p.is_file()}
    if a.mode=='timing':
        frozen=json.loads((base/'frozen.json').read_text())
        for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    out=base/(a.mode+('-'+a.label if a.label else '')+'-process');assert not out.exists();out.mkdir()
    removed=[k for k in os.environ if k.lower().startswith(('lokad_','dotnet_','complus_'))]
    environment={k:v for k,v in os.environ.items() if k not in removed}
    parent=psutil.Process();old_affinity=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(schema=1,mode=a.mode,supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,
        limits=dict(seconds=120,rss=2*1024**3,available=1024**3),binaries=binaries,generation=pin(base/'generation.json'),removed_environment_keys=removed,runs=[])
    save(out/'identity.json',state)
    try:
        for visit in range(1 if a.mode=='proof' else 4):
            folder=out/f'v{visit}';folder.mkdir();child=None
            run=dict(visit=visit,started=time.time(),members={},samples=0,peak_rss=0)
            state['runs'].append(run);save(out/'identity.json',state)
            try:
                with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
                    # Child affinity is inherited before CLR starts; restore only the supervisor.
                    parent.cpu_affinity([2])
                    try:
                        child=subprocess.Popen(['dotnet',str(base/'bin/Probe.dll'),str(model),str(folder/'output'),str(visit),a.mode],
                            stdout=stdout,stderr=stderr,env=environment,**(dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name=='nt' else dict(start_new_session=True)))
                    finally:parent.cpu_affinity([0])
                    birth=psutil.Process(child.pid).create_time();run['child']=dict(pid=child.pid,birth=birth);run['members'][str(child.pid)]=birth
                    start=time.monotonic();save(out/'identity.json',state)
                    while child.poll() is None:
                        members=[]
                        try:
                            owner=psutil.Process(child.pid);assert owner.create_time()==birth
                            for process in [owner]+owner.children(recursive=True):
                                try:
                                    item=dict(pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,affinity=process.cpu_affinity())
                                    assert item['affinity']==[2] and item['birth']>=birth
                                    assert str(item['pid']) not in run['members'] or run['members'][str(item['pid'])]==item['birth']
                                    run['members'][str(item['pid'])]=item['birth'];members.append(item)
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                        samples.write(json.dumps(sample)+'\n');samples.flush();run['samples']+=1
                        run['peak_rss']=max(run['peak_rss'],sum(m['rss'] for m in members));save(out/'identity.json',state)
                        assert sample['seconds']<120 and sample['available']>=1024**3 and run['peak_rss']<2*1024**3,'Resource guard'
                        time.sleep(.25)
                    run['code']=child.wait();run['seconds']=time.monotonic()-start;assert run['code']==0
                    assert json.loads((folder/'output/result.json').read_text())['passed'] is True
            finally:
                if child is not None:
                    for pid,birth in reversed(list(run['members'].items())):
                        try:
                            process=psutil.Process(int(pid))
                            if process.create_time()==birth:process.kill()
                        except psutil.NoSuchProcess:pass
                    child.wait(timeout=10)
                run['ended']=time.time();save(out/'identity.json',state)
        state['complete']=True;state['code']=0
    except BaseException:
        state['error']=traceback.format_exc();state['code']=2;raise
    finally:
        state['ended']=time.time();save(out/'identity.json',state);parent.cpu_affinity(old_affinity)
    print('Completed',a.mode,len(state['runs']),'workers; all observed child births terminal.')

if __name__=='__main__':main()
