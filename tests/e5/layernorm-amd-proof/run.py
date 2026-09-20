"""Two bounded AMD proof processes; this tool performs no latency comparison."""
from pathlib import Path
import argparse,json,os,subprocess,time,traceback
import psutil
from common import FILTER,LIMITS,pin,read,verify

def absent(item):
    try:return psutil.Process(item['pid']).create_time()!=item['birth']
    except psutil.NoSuchProcess:return True

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--payload',type=Path,required=True);args=parser.parse_args();base=args.payload.resolve();bundle=verify(base)
    assert os.name=='posix' and not (base/'result').exists();out=base/'result';out.mkdir()
    parent=psutil.Process();old=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),bundle=pin(base/'bundle.json'),complete=False,limits=LIMITS,runs=[])
    def save():
        temp=out/'identity.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'identity.json')
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))};save()
    try:
        (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text());(out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
        for phase in bundle['phases']:
            folder=out/phase;folder.mkdir();flags={} if phase=='proof' else dict(COMPlus_JitDisasm=FILTER,COMPlus_JitStdOutFile=str(folder/'jit.txt'))
            command=['dotnet',str(base/'bin/LayerNormAmdProof.dll'),phase,str(base/'capture'),str(folder/'output')]
            run=dict(phase=phase,command=command,flags=flags,started=time.time(),members={},samples=0,peak_rss=0);state['runs'].append(run);save();child=None
            try:
                with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
                    assert psutil.virtual_memory().available>=LIMITS['available'];parent.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=base,env=clean|flags,stdout=stdout,stderr=stderr,start_new_session=True)
                    finally:parent.cpu_affinity([0])
                    birth=psutil.Process(child.pid).create_time();run['child']=dict(pid=child.pid,birth=birth);run['members'][str(child.pid)]=birth;start=time.monotonic();save()
                    while child.poll() is None:
                        members=[]
                        try:
                            owner=psutil.Process(child.pid);assert owner.create_time()==birth
                            for process in [owner]+owner.children(recursive=True):
                                try:
                                    value=dict(pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,affinity=process.cpu_affinity())
                                    assert value['affinity']==[2] and (str(value['pid']) not in run['members'] or run['members'][str(value['pid'])]==value['birth'])
                                    run['members'][str(value['pid'])]=value['birth'];members.append(value)
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                        samples.write(json.dumps(sample)+'\n');samples.flush();run['samples']+=1;run['peak_rss']=max(run['peak_rss'],sum(v['rss'] for v in members));save()
                        assert sample['seconds']<LIMITS['seconds'] and sample['available']>=LIMITS['available'] and run['peak_rss']<=LIMITS['rss'],'Resource guard'
                        time.sleep(.1)
                    run['code']=child.wait();run['seconds']=time.monotonic()-start;assert run['code']==0,phase
            finally:
                for pid,birth in reversed(list(run['members'].items())):
                    try:
                        process=psutil.Process(int(pid))
                        if process.create_time()==birth:process.kill()
                    except psutil.NoSuchProcess:pass
                if child is not None:child.wait(timeout=10)
                run['ended']=time.time();run['terminal_members']=all(absent(dict(pid=int(pid),birth=birth)) for pid,birth in run['members'].items());save()
            assert run['terminal_members']
            proof=read(folder/'output/proof.json');assert proof['passed'] and proof['cases']==915 and proof['comparisons']==32182096 and proof['captured']==125
            if phase=='code':assert read(folder/'warmup.json')['passed'] and (folder/'jit.txt').stat().st_size>0
            verify(base);print('Completed',phase,flush=True)
        state['complete']=True;state['code']=0
    except BaseException:state['error']=traceback.format_exc();state['code']=2;raise
    finally:state['ended']=time.time();save();parent.cpu_affinity(old)

if __name__=='__main__':main()
