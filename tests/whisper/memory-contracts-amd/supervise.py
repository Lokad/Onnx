"""One finite AMD contract worker, with affinity inherited before CLR startup."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
from common import pin,read,write
import psutil


def main():
    assert sys.platform=='linux' and psutil.__version__=='7.0.0'
    base=Path(sys.argv[1]);frozen=read(base/'frozen.json');limits=frozen['limits']
    assert not (base/'run').exists();folder=base/'run';folder.mkdir()
    parent=psutil.Process();parent.cpu_affinity([0]);start=time.monotonic()
    state=dict(complete=False,code=None,started=time.time(),frozen=pin(base/'frozen.json'),limits=limits,
        supervisor=dict(pid=parent.pid,birth=parent.create_time()),members={},samples=0,peak_rss=0)
    def save():
        path=folder/'identity.tmp';path.write_text(json.dumps(state,indent=2));path.replace(folder/'identity.json')
    save();child=None
    try:
        for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
        for name,wanted in frozen['external'].items():assert pin(name)==wanted,name
        available=psutil.virtual_memory().available;disk=psutil.disk_usage(str(base)).free
        state.update(preflight_available=available,preflight_disk=disk);save()
        write(folder/'preflight.json',dict(available=available,disk=disk,required_available=limits['preflight'],required_disk=limits['preflight_disk']))
        assert available>=limits['preflight'] and disk>=limits['preflight_disk']
        env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        command=[frozen['managed_runtime']['host'],str(base/'bin/WhisperMemoryContractsV2.dll'),frozen['models'],
            str(base/'inputs/inputs.json'),str(base/'short/manifest.json'),str(folder/'worker')]
        state['command']=command;execution=time.monotonic();save()
        with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
            parent.cpu_affinity([2])
            try:child=subprocess.Popen(command,cwd=base,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
            finally:parent.cpu_affinity([0])
            process=psutil.Process(child.pid);state['child']=dict(pid=child.pid,birth=process.create_time())
            state['members'][str(child.pid)]=process.create_time();save()
            while child.poll() is None:
                members=[]
                try:
                    assert process.create_time()==state['child']['birth']
                    for member in [process]+process.children(recursive=True):
                        try:
                            birth=member.create_time();assert state['members'].get(str(member.pid),birth)==birth
                            state['members'][str(member.pid)]=birth;threads=[]
                            for thread in member.threads():
                                try:threads.append(dict(tid=thread.id,affinity=sorted(os.sched_getaffinity(thread.id))))
                                except ProcessLookupError:pass
                            members.append(dict(pid=member.pid,birth=birth,rss=member.memory_info().rss,affinity=member.cpu_affinity(),threads=threads))
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                if not members and child.poll() is not None:break
                row=dict(seconds=time.monotonic()-execution,members=members,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(base)).free)
                samples.write(json.dumps(row)+'\n');samples.flush();state['samples']+=1
                state['peak_rss']=max(state['peak_rss'],sum(m['rss'] for m in members));save()
                assert 0<row['seconds']<limits['seconds'] and members and row['available']>=limits['available'] and row['disk']>=limits['disk']
                assert sum(m['rss'] for m in members)<limits['rss']
                assert all(m['affinity']==[2] and m['threads'] and all(t['affinity']==[2] for t in m['threads']) for m in members)
                time.sleep(.5)
            state['code']=child.wait();state['execution_seconds']=time.monotonic()-execution;save();assert state['code']==0
        result=read(folder/'worker/result.json')
        assert result['runtime']=='.NET 10.0.8' and result['affinity']==4 and result['processor_count']==1 and result['flags']=={}
        assert result['refusals']==16 and len(result['concurrent_speech'])==2
        for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
        for name,wanted in frozen['external'].items():assert pin(name)==wanted,name
        state['result']=pin(folder/'worker/result.json')
    except BaseException as error:
        state['error']=repr(error)
        for pid,birth in reversed(list(state['members'].items())):
            try:
                process=psutil.Process(int(pid))
                if process.create_time()==birth:process.kill()
            except psutil.NoSuchProcess:pass
        if child is not None:child.wait(timeout=10)
        traceback.print_exc();raise
    finally:
        state.update(complete=True,ended=time.time(),seconds=time.monotonic()-start);save()


if __name__=='__main__':main()
