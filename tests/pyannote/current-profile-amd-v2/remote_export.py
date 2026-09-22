"""Convert the two terminal immutable captures on AMD under the original export bounds."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from remote import BASE,DOTNET,LIMITS,artifact_size,clean_env,idle,live,pin,read,save,thread_affinities,verify
import psutil

JOBS=[(name,format) for name in ['sampled-a','sampled-b'] for format in ['Speedscope','Chromium']]


def main():
    own=psutil.Process();own.cpu_affinity([0]);idle();verify()
    assert psutil.boot_time()==1789634288.0
    receipt=read(BASE/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    for name,wanted in receipt['files'].items():assert pin(BASE/name)==wanted,name
    output=BASE/'exports';assert not output.exists();output.mkdir();(output/'logs').mkdir()
    state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),boot_time=psutil.boot_time(),started=time.time(),runs=[])
    path=output/'identity.json';save(path,state)
    try:
        for capture,format in JOBS:
            name=capture+'-'+format.lower();verify()
            assert pin(BASE/capture/'capture.nettrace')==receipt['files'][capture+'/capture.nettrace']
            assert psutil.virtual_memory().available>=8*1024**3 and psutil.disk_usage(BASE).free>=3*1024**3
            destination=output/capture;destination.mkdir(exist_ok=True)
            row=dict(name=name,complete=False,code=None,preflight=dict(available=psutil.virtual_memory().available,disk=psutil.disk_usage(BASE).free),samples=0,peak_rss=0)
            state['runs'].append(row);save(path,state);child=None;start=time.monotonic()
            try:
                command=[DOTNET,BASE/'tracer/dotnet-trace.dll','convert',BASE/capture/'capture.nettrace','--format',format,'--output',destination/format.lower()]
                row['command']=list(map(str,command))
                with (output/'logs'/(name+'.stdout')).open('x') as stdout,(output/'logs'/(name+'.stderr')).open('x') as stderr,(output/'logs'/(name+'.jsonl')).open('x') as log:
                    child=subprocess.Popen(list(map(str,command)),cwd=BASE,env=clean_env(),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                    process=psutil.Process(child.pid);row['identity']=dict(pid=child.pid,birth=process.create_time());save(path,state)
                    while child.poll() is None:
                        members=[]
                        try:
                            assert process.create_time()==row['identity']['birth'] and not process.children(recursive=True)
                            members.append(dict(pid=process.pid,birth=process.create_time(),affinity=process.cpu_affinity(),threads=thread_affinities(process),rss=process.memory_info().rss))
                        except psutil.NoSuchProcess:pass
                        sample=dict(seconds=time.monotonic()-start,members=members,rss=sum(m['rss'] for m in members),available=psutil.virtual_memory().available,
                            disk=psutil.disk_usage(BASE).free,output_bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),artifacts=artifact_size())
                        log.write(json.dumps(sample)+'\n');log.flush();row['samples']+=1;row['peak_rss']=max(row['peak_rss'],sample['rss']);save(path,state)
                        assert sample['seconds']<900 and sample['rss']<8*1024**3 and sample['available']>=1024**3 and sample['disk']>=1024**3
                        assert sample['output_bytes']<=1024**3 and sample['artifacts']<=LIMITS['artifacts']
                        assert all(m['affinity']==[0] and all(t['affinity']==[0] for t in m['threads']) for m in members)
                        time.sleep(.25)
                    row['code']=child.wait();assert row['code']==0 and not live(row['identity'])
                extension='speedscope' if format=='Speedscope' else 'chromium'
                generated=destination/(format.lower()+'.'+extension+'.json');assert generated.is_file() and generated.stat().st_size>0
                row['output']=dict(path=generated.relative_to(output).as_posix(),**pin(generated))
            except BaseException:
                row['error']=traceback.format_exc()
                if child is not None and child.poll() is None:
                    if 'identity' not in row or live(row['identity']):child.kill()
                if child is not None:child.wait(timeout=15)
                raise
            finally:
                row.update(complete=True,code=None if child is None else child.poll(),seconds=time.monotonic()-start);save(path,state)
            print(name,'exported',flush=True)
        verify()
        for name,wanted in receipt['files'].items():assert pin(BASE/name)==wanted,name
        state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());traceback.print_exc()
    finally:
        state.update(complete=True,ended=time.time());save(path,state)
    return state['code']


if __name__=='__main__':raise SystemExit(main())
