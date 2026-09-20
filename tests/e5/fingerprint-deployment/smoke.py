"""Bound local public-policy/native smokes; preserve every original process and array."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil
from audit import pin,read,write,worker,telemetry
from protocol import schedule,LIMITS

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);args=p.parse_args();base=args.artifact.resolve()
    shutil.copytree(ROOT/'artifacts/e5-fingerprint-product-v2-20260920/payload/inputs',base/'inputs')
    out=base/'smoke-process';out.mkdir();model=ROOT/'models/multilingual-e5-small/model.onnx';native=ROOT/'artifacts/e5-public-ort-20260919/bin/onnxruntime.dll'
    jobs=[]
    for phase,policy,role,index in [('aa','default',r,0) for r in ['A','B','C','N']]+[('aa','memory',r,0) for r in ['A','N']]+[('compare',p,'C',0) for p in ['default','memory']]+[('aa','default','N',4),('compare','memory','C',4),('aa','memory','B',1),('compare','default','C',2),('aa','default','A',3)]:
        job=next(j.copy() for j in schedule() if j['visit']==0 and j['policy']==policy and j['role']==role and j['case_index']==index)
        job['name']=phase+'-'+job['name'];job['phase']=phase;jobs.append(job)
    parent=psutil.Process();old=parent.cpu_affinity();parent.cpu_affinity([0])
    state=dict(supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,limits=LIMITS,runs=[])
    def save():
        tmp=out/'identity.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'identity.json')
    clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))};results=[];save()
    source={p.name:pin(p) for p in Path(__file__).parent.iterdir() if p.is_file()}
    binaries={p.name:pin(p) for p in (base/'bin').iterdir() if p.is_file()}
    try:
        for job in jobs:
            folder=out/job['name'];folder.mkdir();phase=job['phase'];child=None
            command=['dotnet',str(base/'bin/FingerprintDeployment.dll'),str(model),str(base/'inputs'/(job['case']+'.json')),str(folder/'output'),
                     job['policy'],job['role'],phase,str(job['visit']),str(job['case_index']),str(native),'smoke']
            run=dict(job=job,command=command,started=time.time(),members={},samples=0,peak_rss=0);state['runs'].append(run);save()
            try:
                with (folder/'stdout.txt').open('x') as stdout,(folder/'stderr.txt').open('x') as stderr,(folder/'samples.jsonl').open('x') as samples:
                    parent.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=ROOT,env=clean|({'LOKAD_ONNX_FINGERPRINT_STRINGS':'1'} if phase=='compare' and job['role']=='C' else {}),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                    finally:parent.cpu_affinity([0])
                    birth=psutil.Process(child.pid).create_time();run['child']=dict(pid=child.pid,birth=birth);run['members'][str(child.pid)]=birth;start=time.monotonic();save()
                    while child.poll() is None:
                        members=[]
                        try:
                            owner=psutil.Process(child.pid);assert owner.create_time()==birth
                            for process in [owner]+owner.children(recursive=True):
                                try:
                                    row=dict(pid=process.pid,birth=process.create_time(),rss=process.memory_info().rss,affinity=process.cpu_affinity());assert row['affinity']==[2]
                                    assert str(row['pid']) not in run['members'] or run['members'][str(row['pid'])]==row['birth']
                                    run['members'][str(row['pid'])]=row['birth'];members.append(row)
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,members=members)
                        samples.write(json.dumps(sample)+'\n');samples.flush();run['samples']+=1;run['peak_rss']=max(run['peak_rss'],sum(r['rss'] for r in members));save()
                        assert sample['seconds']<300 and sample['available']>=1024**3 and run['peak_rss']<6*1024**3,'Resource guard';time.sleep(.25)
                    run['code']=child.wait();run['seconds']=time.monotonic()-start;assert run['code']==0,job
            finally:
                if child is not None:
                    for pid,birth in reversed(list(run['members'].items())):
                        try:
                            process=psutil.Process(int(pid))
                            if process.create_time()==birth:process.kill()
                        except psutil.NoSuchProcess:pass
                    child.wait(timeout=10)
                run['ended']=time.time();save()
            value=worker(folder/'output',base/'inputs',pin(model),binaries['FingerprintDeployment.dll'],binaries['Microsoft.ML.OnnxRuntime.dll'],pin(native),job,phase,True)
            results.append(dict(job=job,measured=len(value['measured']),error=value['after_error'],output_sha256=value['output_sha256'],cache_entries=value['cache_entries']))
            print('Passed',job['name'],flush=True)
        state['complete']=True;state['code']=0
    except BaseException:state['error']=traceback.format_exc();state['code']=2;raise
    finally:state['ended']=time.time();save();parent.cpu_affinity(old)
    samples={j['name']:[json.loads(line) for line in (out/j['name']/'samples.jsonl').read_text().splitlines()] for j in jobs}
    resources=telemetry(state,samples,jobs)
    for item in resources['births']:
        if item['pid']==parent.pid:continue
        try:assert psutil.Process(item['pid']).create_time()!=item['birth']
        except psutil.NoSuchProcess:pass
    write(base/'smoke-audit.json',dict(passed=True,jobs=jobs,results=results,resources=resources,source=source,binaries=binaries,model=pin(model),native=pin(native)))
    print('All thirteen functional smokes pass; no performance verdict.')

if __name__=='__main__':main()
