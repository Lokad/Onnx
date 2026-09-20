"""Only one independent engine is resumed at a time; retain all binary commands."""
from pathlib import Path
import argparse,hashlib,json,os,queue,struct,subprocess,sys,threading,time,traceback
import psutil
from protocol import schedule,commands,LIMITS,CRITERIA

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def read(path):return json.loads(path.read_text())
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def terminal(items):
    for item in items:
        try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
        except psutil.NoSuchProcess:pass

def cohort(base,out,job,phase,model,native,smoke,state,save_state):
    folder=out/job['name'];folder.mkdir();parent=psutil.Process();workers={}
    record=dict(job=job,started=time.time(),workers={},events=[],samples=0,peak_rss=0,complete=False)
    state['runs'].append(record);save_state();start=time.monotonic()
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    samples=(folder/'samples.jsonl').open('x');active=None
    if not smoke:
        import campaign_processes as accounting
        pre=accounting.snapshot();save(folder/'pre.json',pre);(folder/'cpu-before.txt').write_text(Path('/proc/stat').read_text())

    def original(role):
        w=workers[role];p=psutil.Process(w['child'].pid)
        assert p.create_time()==w['birth'],('Worker birth changed',role,w['birth'],p.create_time())
        descendants=[dict(pid=c.pid,birth=c.create_time(),name=c.name()) for c in p.children(recursive=True)]
        assert not descendants,('Unexpected worker descendants',role,descendants)
        return p

    def snapshot():
        members=[]
        for role,w in workers.items():
            if w['child'].poll() is not None:continue
            try:
                p=original(role);cpu=p.cpu_times();status=p.status()
                item=dict(role=role,pid=p.pid,birth=p.create_time(),rss=p.memory_info().rss,affinity=p.cpu_affinity(),
                    status=status,suspended=w['suspended'],cpu_seconds=cpu.user+cpu.system)
            except psutil.NoSuchProcess:continue
            assert item['affinity']==[2]
            if role!=active:
                assert w['suspended']
                if sys.platform.startswith('linux'):assert status==psutil.STATUS_STOPPED,('Inactive worker running',role,status)
                assert abs(item['cpu_seconds']-w['paused_cpu'])<=.02,('Inactive CPU advanced',role)
            members.append(item)
        value=dict(seconds=time.monotonic()-start,active=active,available=psutil.virtual_memory().available,members=members)
        samples.write(json.dumps(value)+'\n');samples.flush();record['samples']+=1
        record['peak_rss']=max(record['peak_rss'],sum(p['rss'] for p in members))
        assert value['seconds']<LIMITS['seconds'] and value['available']>=LIMITS['available'] and record['peak_rss']<LIMITS['rss'],'Resource guard'
        return value

    def reader(stream,messages):
        try:
            while True:
                packet=stream.read(8)
                if not packet:messages.put(None);return
                if len(packet)!=8:raise IOError('Truncated acknowledgement')
                messages.put(packet)
        except BaseException as error:messages.put(error)

    def wait_ack(role,op,index):
        w=workers[role]
        while True:
            snapshot()
            try:value=w['messages'].get(timeout=.25)
            except queue.Empty:
                assert w['child'].poll() is None,('Worker exited before acknowledgement',role,w['child'].returncode)
                continue
            assert isinstance(value,bytes) and value==struct.pack('<ii',op,index),('Invalid acknowledgement',role,value,op,index)
            break

    def pause(role):
        w=workers[role];p=original(role);p.suspend()
        if sys.platform.startswith('linux'):
            until=time.monotonic()+2
            while p.status()!=psutil.STATUS_STOPPED:
                assert time.monotonic()<until,'Worker did not stop';time.sleep(.001)
        cpu=p.cpu_times();w['paused_cpu']=cpu.user+cpu.system;w['suspended']=True

    try:
        for role,op,index in commands(job,smoke):
            assert active is None
            event=dict(role=role,op=op,index=index,started=time.monotonic()-start)
            record['events'].append(event)
            if op==0:
                destination=folder/role;destination.mkdir();stderr=(destination/'stderr.txt').open('x')
                command=['dotnet',str(base/'bin/InterleavedProcesses.dll'),str(model),str(base/'inputs'/(job['case']+'.json')),
                    str(destination/'output'),job['policy'],role,phase,str(job['visit']),str(job['case_index']),str(native),'smoke' if smoke else 'full']
                env=clean|({'LOKAD_ONNX_FINGERPRINT_STRINGS':'1','LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT':'1'} if phase=='compare' and role=='C' else {})
                parent.cpu_affinity([2])
                try:child=subprocess.Popen(command,cwd=base,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                    **(dict(creationflags=subprocess.DETACHED_PROCESS) if os.name=='nt' else dict(start_new_session=True)))
                finally:parent.cpu_affinity([0])
                birth=psutil.Process(child.pid).create_time();messages=queue.Queue()
                thread=threading.Thread(target=reader,args=(child.stdout,messages),daemon=True);thread.start()
                workers[role]=dict(child=child,birth=birth,messages=messages,thread=thread,stderr=stderr,suspended=False,paused_cpu=0.)
                record['workers'][role]=dict(pid=child.pid,birth=birth,command=command,started=time.time())
                active=role;save_state();wait_ack(role,op,index)
            else:
                w=workers[role];original(role);assert w['suspended']
                snapshot();active=role;w['suspended']=False;original(role).resume()
                w['child'].stdin.write(struct.pack('<ii',op,index));w['child'].stdin.flush();save_state();wait_ack(role,op,index)
            if op==4:
                w=workers[role]
                while w['child'].poll() is None:snapshot();time.sleep(.01)
                assert w['child'].returncode==0,('Nonzero exit',role)
                w['thread'].join(timeout=2);assert not w['thread'].is_alive()
                w['stderr'].close();w['child'].stdin.close();w['child'].stdout.close()
                record['workers'][role].update(code=0,ended=time.time())
            else:pause(role)
            active=None;event['ended']=time.monotonic()-start;event['ack']=[op,index]
            snapshot();save_state()
        record['complete']=True;record['code']=0
    except BaseException:
        record['error']=traceback.format_exc();record['code']=2;raise
    finally:
        for role,w in workers.items():
            if w['child'].poll() is None:
                # Cleanup must not depend on the invariant that triggered failure.
                try:
                    p=psutil.Process(w['child'].pid)
                    if p.create_time()==w['birth']:
                        descendants=[(c,c.create_time()) for c in p.children(recursive=True)]
                        record['workers'][role]['cleanup_descendants']=[dict(pid=c.pid,birth=b) for c,b in descendants]
                        for child,birth in reversed(descendants):
                            try:
                                if birth>=w['birth'] and child.create_time()==birth:child.kill()
                            except psutil.NoSuchProcess:pass
                        p.kill()
                except psutil.NoSuchProcess:pass
                w['child'].wait(timeout=10)
            record['workers'][role].update(code=w['child'].returncode,ended=record['workers'][role].get('ended',time.time()))
            w['stderr'].close();w['child'].stdin.close();w['child'].stdout.close();w['thread'].join(timeout=2)
        samples.close();record['seconds']=time.monotonic()-start;record['ended']=time.time()
        terminal(record['workers'].values())
        if not smoke:
            post=accounting.snapshot();save(folder/'post.json',post);(folder/'cpu-after.txt').write_text(Path('/proc/stat').read_text())
            record['accounting']=accounting.foreign_fraction(pre,post,parent.pid)
        save_state()
    print('Complete',phase,job['name'],round(record['seconds'],3),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True)
    p.add_argument('--gate-sha256');a=p.parse_args();base=a.payload.resolve();meta=read(base/'frozen.json')
    assert sys.platform.startswith('linux') and meta['schedule']==schedule() and meta['limits']==LIMITS and meta['criteria']==CRITERIA
    for name,wanted in meta['files'].items():assert pin(base/name)==wanted,name
    for asset in ['model','native']:assert pin(Path(meta[asset]['path']))=={k:meta[asset][k] for k in ['bytes','sha256']}
    if a.phase=='compare':
        gate=read(base/'aa-gate.json');assert pin(base/'aa-gate.json')['sha256']==a.gate_sha256
        assert gate['passed'] and gate['timing_passed'] and gate['frozen']==pin(base/'frozen.json')
        for name,wanted in gate['remote_files'].items():assert pin(base/name)==wanted,name
        terminal(gate['births'])
    else:assert a.gate_sha256 is None
    out=base/('result-'+a.phase);assert not out.exists();out.mkdir()
    parent=psutil.Process();old=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(phase=a.phase,supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,
        frozen=pin(base/'frozen.json'),limits=LIMITS,gate_sha256=a.gate_sha256,runs=[])
    def save_state():save(out/'identity.json',state)
    save_state()
    try:
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],cwd=base,text=True))
        (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
        for job in meta['schedule']:cohort(base,out,job,a.phase,Path(meta['model']['path']),Path(meta['native']['path']),False,state,save_state)
        state['complete']=True;state['code']=0
    except BaseException:state['code']=2;state['error']=traceback.format_exc();raise
    finally:state['ended']=time.time();save_state();parent.cpu_affinity(old)

if __name__=='__main__':main()
