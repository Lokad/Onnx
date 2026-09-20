"""Four sequential complete-bank timing workers with retained process accounting."""
from pathlib import Path
import argparse,json,os,subprocess,time,traceback
import psutil
import campaign_processes as accounting
from common import LIMITS,pin,read,verify

def terminal(items):
    for item in items:
        try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
        except psutil.NoSuchProcess:pass

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--payload',type=Path,required=True);parser.add_argument('--origin',type=Path,required=True);args=parser.parse_args()
    base=args.payload.resolve();origin=args.origin.resolve();verify(base,origin)
    assert os.name=='posix' and not (base/'result').exists();out=base/'result';out.mkdir()
    parent=psutil.Process();old=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),bundle=pin(base/'bundle.json'),complete=False,limits=LIMITS,runs=[])
    def save():
        temporary=out/'identity.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(out/'identity.json')
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))};save()
    try:
        (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text());(out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
        for visit in range(4):
            folder=out/str(visit);folder.mkdir();child=None
            command=['dotnet',str(base/'bin/LayerNormBank.dll'),'run',str(origin),str(base/'banks.json'),str(folder/'output'),str(visit)]
            run=dict(visit=visit,command=command,started=time.time(),members={},samples=0,peak_rss=0);state['runs'].append(run);save()
            pre=accounting.snapshot();(folder/'pre.json').write_text(json.dumps(pre));(folder/'cpu-before.txt').write_text(Path('/proc/stat').read_text())
            try:
                with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
                    assert psutil.virtual_memory().available>=LIMITS['available'];parent.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=base,env=clean,stdout=stdout,stderr=stderr,start_new_session=True)
                    finally:parent.cpu_affinity([0])
                    birth=psutil.Process(child.pid).create_time();run['child']=dict(pid=child.pid,birth=birth);run['members'][str(child.pid)]=birth;start=time.monotonic();save()
                    while child.poll() is None:
                        members=[]
                        try:
                            owner=psutil.Process(child.pid);assert owner.create_time()==birth
                            for process in [owner]+owner.children(recursive=True):
                                try:
                                    cpu=process.cpu_times();value=dict(pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,affinity=process.cpu_affinity(),cpu=cpu.user+cpu.system)
                                    assert value['affinity']==[2] and value['birth']>=birth
                                    assert str(value['pid']) not in run['members'] or run['members'][str(value['pid'])]==value['birth']
                                    run['members'][str(value['pid'])]=value['birth'];members.append(value)
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                        samples.write(json.dumps(sample)+'\n');samples.flush();run['samples']+=1;run['peak_rss']=max(run['peak_rss'],sum(m['rss'] for m in members));save()
                        assert sample['seconds']<LIMITS['seconds'] and sample['available']>=LIMITS['available'] and run['peak_rss']<=LIMITS['rss'],'Resource guard'
                        time.sleep(.25)
                    run['code']=child.wait();run['seconds']=time.monotonic()-start;assert run['code']==0,visit
            finally:
                for pid,birth in reversed(list(run['members'].items())):
                    try:
                        process=psutil.Process(int(pid))
                        if process.create_time()==birth:process.kill()
                    except psutil.NoSuchProcess:pass
                if child is not None:child.wait(timeout=10)
                run['ended']=time.time();terminal([dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items()]);run['terminal_members']=True;save()
            post=accounting.snapshot();(folder/'post.json').write_text(json.dumps(post));(folder/'cpu-after.txt').write_text(Path('/proc/stat').read_text())
            run['accounting']=accounting.foreign_fraction(pre,post,parent.pid);save()
            assert run['accounting']['foreign_cpu_fraction']<=LIMITS['foreign'],'Foreign CPU guard'
            assert read(folder/'output/timing.json')['visit']==visit
            verify(base,origin);print('Completed visit',visit,flush=True)
        state['complete']=True;state['code']=0
    except BaseException:state['error']=traceback.format_exc();state['code']=2;raise
    finally:state['ended']=time.time();save();parent.cpu_affinity(old)

if __name__=='__main__':main()
