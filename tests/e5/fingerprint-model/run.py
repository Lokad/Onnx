"""Fixed single-graph timing phases, with a preceding independently audited A/A gate."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time,traceback
import psutil
import campaign_processes as accounting

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def read(path):return json.loads(path.read_text())
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def terminal(births):
    for item in births:
        try:assert psutil.Process(item['pid']).create_time()!=item['birth'],('Owned process live',item)
        except psutil.NoSuchProcess:pass

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True)
    p.add_argument('--gate-sha256');a=p.parse_args();base=a.payload.resolve();meta=read(base/'frozen.json');phase=a.phase
    for name,wanted in meta['files'].items():assert pin(base/name)==wanted,name
    assert pin(Path(meta['model']['path']))=={k:meta['model'][k] for k in ['bytes','sha256']}
    if phase=='compare':
        assert a.gate_sha256 and pin(base/'aa-gate.json')['sha256']==a.gate_sha256
        gate=read(base/'aa-gate.json');assert gate['passed'] is True and gate['timing_passed'] is True and gate['frozen']==pin(base/'frozen.json')
        prior=read(base/'result-aa/identity.json');assert prior['complete'] is True and prior['code']==0
        assert gate['identity']==pin(base/'result-aa/identity.json')
        for name,wanted in gate['remote_files'].items():assert pin(base/name)==wanted,name
        terminal(gate['births'])
    else:assert a.gate_sha256 is None
    out=base/('result-'+phase);assert not out.exists();out.mkdir()
    parent=psutil.Process();old=parent.cpu_affinity();parent.cpu_affinity([0])
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    state=dict(phase=phase,supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,
               limits=meta['limits'],frozen=pin(base/'frozen.json'),gate_sha256=a.gate_sha256,runs=[])
    save(out/'identity.json',state)
    try:
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],cwd=base,text=True))
        (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
        for job in meta['schedule']:
            folder=out/job['name'];folder.mkdir();child=None
            command=['dotnet',str(base/'bin/FingerprintModel.dll'),meta['model']['path'],str(base/'inputs'/(job['case']+'.json')),
                     str(folder/'output'),str(job['visit']),str(job['case_index']),phase]
            run=dict(job=job,command=command,started=time.time(),members={},samples=0,peak_rss=0)
            state['runs'].append(run);save(out/'identity.json',state)
            pre=accounting.snapshot();(folder/'pre.json').write_text(json.dumps(pre));(folder/'cpu-before.txt').write_text(Path('/proc/stat').read_text())
            try:
                with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
                    parent.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=base,env=clean|{'LOKAD_ONNX_FINGERPRINT_STRINGS':'1'},stdout=stdout,stderr=stderr,start_new_session=True)
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
                        assert sample['seconds']<300 and sample['available']>=1024**3 and run['peak_rss']<6*1024**3,'Resource guard'
                        time.sleep(.5)
                    run['code']=child.wait();run['seconds']=time.monotonic()-start;assert run['code']==0,job
            finally:
                if child is not None:
                    for pid,birth in reversed(list(run['members'].items())):
                        try:
                            process=psutil.Process(int(pid))
                            if process.create_time()==birth:process.kill()
                        except psutil.NoSuchProcess:pass
                    child.wait(timeout=10)
                run['ended']=time.time();save(out/'identity.json',state)
            post=accounting.snapshot();(folder/'post.json').write_text(json.dumps(post));(folder/'cpu-after.txt').write_text(Path('/proc/stat').read_text())
            run['accounting']=accounting.foreign_fraction(pre,post,parent.pid);save(out/'identity.json',state)
            print('Complete',phase,job['name'],run['seconds'],flush=True)
        state['complete']=True;state['code']=0
    except BaseException:
        state['error']=traceback.format_exc();state['code']=2;raise
    finally:
        state['ended']=time.time();save(out/'identity.json',state);parent.cpu_affinity(old)

if __name__=='__main__':main()
