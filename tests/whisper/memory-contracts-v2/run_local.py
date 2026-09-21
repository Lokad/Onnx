"""Finite normal-runtime Windows contract run, with inherited CPU affinity."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time,traceback
from prepare import ROOT,BASE,PRIOR,pin,read,write
sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

LIMITS=dict(seconds=1800,rss=14*1024**3,available=1024**3,preflight=13*1024**3,disk=32*1024**2,preflight_disk=64*1024**2)


def main():
    assert psutil.__version__=='7.0.0' and os.name=='nt'
    prepared=read(BASE/'prepared.json');assert prepared['prepared']
    for name,wanted in prepared['files'].items():assert pin(BASE/name)==wanted,name
    for name,wanted in prepared['inputs'].items():assert pin(ROOT/name)==wanted,name
    folder=BASE/'local';folder.mkdir();shutil.copyfile(Path(__file__),folder/'run_local.py')
    runtime=Path('C:/Program Files/dotnet/shared/Microsoft.NETCore.App/10.0.12')
    external={str(p):pin(p) for p in sorted(runtime.rglob('*')) if p.is_file()}
    dotnet=Path(shutil.which('dotnet'));external[str(dotnet)]=pin(dotnet)
    frozen=dict(prepared=pin(BASE/'prepared.json'),external=external,limits=LIMITS,supervisor=pin(folder/'run_local.py'),
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(folder/'frozen.json',frozen)
    available=psutil.virtual_memory().available;disk=psutil.disk_usage(str(BASE)).free
    write(folder/'preflight.json',dict(available=available,disk=disk,required_available=LIMITS['preflight'],required_disk=LIMITS['preflight_disk']))
    assert available>=LIMITS['preflight'] and disk>=LIMITS['preflight_disk']
    parent=psutil.Process();prior_affinity=parent.cpu_affinity()
    state=dict(complete=False,code=None,started=time.time(),supervisor=dict(pid=parent.pid,birth=parent.create_time()),
        frozen=pin(folder/'frozen.json'),limits=LIMITS,preflight_available=available,preflight_disk=disk,samples=0,peak_rss=0,members={})
    def save():
        path=folder/'identity.tmp';path.write_text(json.dumps(state,indent=2));path.replace(folder/'identity.json')
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    command=[str(dotnet),str(BASE/'bin/WhisperMemoryContractsV2.dll'),str(ROOT/'models/whisper-large-v3-turbo'),
        str(PRIOR/'inputs/inputs.json'),str(ROOT/'artifacts/asr-labeled-20260919/native-whisper/manifest.json'),str(folder/'worker')]
    state['command']=command;child=None;start=time.monotonic();save()
    try:
        with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
            parent.cpu_affinity([2])
            try:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            finally:parent.cpu_affinity(prior_affinity)
            process=psutil.Process(child.pid);state['child']=dict(pid=child.pid,birth=process.create_time());state['members'][str(child.pid)]=process.create_time();save()
            while child.poll() is None:
                members=[]
                try:
                    assert process.create_time()==state['child']['birth']
                    for member in [process]+process.children(recursive=True):
                        try:
                            birth=member.create_time();assert state['members'].get(str(member.pid),birth)==birth
                            state['members'][str(member.pid)]=birth
                            members.append(dict(pid=member.pid,birth=birth,rss=member.memory_info().rss,affinity=member.cpu_affinity()))
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                if not members and child.poll() is not None:break
                row=dict(seconds=time.monotonic()-start,members=members,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(BASE)).free)
                samples.write(json.dumps(row)+'\n');samples.flush();state['samples']+=1;state['peak_rss']=max(state['peak_rss'],sum(m['rss'] for m in members));save()
                assert 0<=row['seconds']<LIMITS['seconds'] and members
                assert row['available']>=LIMITS['available'] and row['disk']>=LIMITS['disk']
                assert sum(m['rss'] for m in members)<LIMITS['rss'] and all(m['affinity']==[2] for m in members)
                time.sleep(.5)
            state['code']=child.wait();assert state['code']==0
        result=read(folder/'worker/result.json');assert result['runtime']=='.NET 10.0.12' and result['affinity']==4 and result['processor_count']==1 and result['flags']=={}
        assert result['refusals']==16 and len(result['concurrent_speech'])==2
        for name,wanted in prepared['files'].items():assert pin(BASE/name)==wanted,name
        for name,wanted in prepared['inputs'].items():assert pin(ROOT/name)==wanted,name
        for name,wanted in external.items():assert pin(Path(name))==wanted,name
        state['result']=pin(folder/'worker/result.json')
    except BaseException as error:
        state['error']=repr(error)
        for pid,birth in reversed(list(state['members'].items())):
            try:
                target=psutil.Process(int(pid))
                if target.create_time()==birth:target.kill()
            except psutil.NoSuchProcess:pass
        if child is not None:child.wait(timeout=10)
        traceback.print_exc();raise
    finally:
        state.update(complete=True,ended=time.time(),seconds=time.monotonic()-start);save()
    print(json.dumps(dict(passed=True,result=state['result'],samples=state['samples'],peak_rss=state['peak_rss'])))


if __name__=='__main__':main()
