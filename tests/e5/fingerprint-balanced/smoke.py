"""One bounded local functional replay, with actual PID/birth and child-only environment cleanup."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil
from audit import pin,read,write,worker

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    inputs=base/'inputs';assert not inputs.exists();shutil.copytree(ROOT/'artifacts/e5-fingerprint-product-v2-20260920/payload/inputs',inputs)
    out=base/'smoke-process';assert not out.exists();out.mkdir();model=ROOT/'models/multilingual-e5-small/model.onnx'
    parent=psutil.Process();old=parent.cpu_affinity();child=None
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}|{'LOKAD_ONNX_FINGERPRINT_STRINGS':'1'}
    state=dict(supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,limits=dict(seconds=60,rss=6*1024**3,available=1024**3),members={},samples=0,peak_rss=0)
    def save():(out/'identity.json').write_text(json.dumps(state,indent=2))
    try:
        with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr,(out/'samples.jsonl').open('x') as samples:
            parent.cpu_affinity([2])
            try:child=subprocess.Popen(['dotnet',str(base/'bin/FingerprintBalanced.dll'),str(model),str(inputs/'e5-8tok.json'),str(out/'output'),'0','0','smoke'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            finally:parent.cpu_affinity([0])
            birth=psutil.Process(child.pid).create_time();state['child']=dict(pid=child.pid,birth=birth);state['members'][str(child.pid)]=birth;save();start=time.monotonic()
            while child.poll() is None:
                members=[]
                try:
                    owner=psutil.Process(child.pid);assert owner.create_time()==birth
                    for item in [owner]+owner.children(recursive=True):
                        try:
                            row=dict(pid=item.pid,birth=item.create_time(),rss=item.memory_info().rss,affinity=item.cpu_affinity());assert row['affinity']==[2]
                            state['members'][str(item.pid)]=row['birth'];members.append(row)
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                samples.write(json.dumps(sample)+'\n');samples.flush();state['samples']+=1;state['peak_rss']=max(state['peak_rss'],sum(r['rss'] for r in members));save()
                assert sample['seconds']<60 and sample['available']>=1024**3 and state['peak_rss']<6*1024**3,'Resource guard';time.sleep(.25)
            state['code']=child.wait();assert state['code']==0
            value=worker(out/'output',inputs,pin(model),pin(base/'bin/FingerprintBalanced.dll'),'smoke',0,0)
            write(base/'smoke-audit.json',dict(passed=True,model=pin(model),binaries={p.name:pin(p) for p in (base/'bin').iterdir() if p.is_file()},
                source={p.name:pin(p) for p in Path(__file__).parent.iterdir() if p.suffix in ['.py','.cs','.csproj']},
                measured=len(value['measured']),native_error=value['after_error'],cache_entries=value['cache_entries']))
            state['complete']=True
    except BaseException:state['error']=traceback.format_exc();raise
    finally:
        if child is not None:
            for pid,birth in reversed(list(state['members'].items())):
                try:
                    process=psutil.Process(int(pid))
                    if process.create_time()==birth:process.kill()
                except psutil.NoSuchProcess:pass
            child.wait(timeout=10)
        state['ended']=time.time();save();parent.cpu_affinity(old)
    print('Local functional smoke passed; no AMD timing claim.')

if __name__=='__main__':main()
