"""Bounded local input verification followed by two complete meetings and recovery."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from prepare import ROOT, BASE, pin, save, clean_env
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def absent(identity):
    try:return psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess:return True


def main():
    assert not (BASE/'processes.json').exists()
    spec=json.loads((BASE/'prepared.json').read_text());manifest=json.loads((BASE/'manifest.json').read_text());limits=manifest['limits']
    controls=dict(prepared=pin(BASE/'prepared.json'),runner=pin(Path(__file__)),limits=limits)
    save(BASE/'run-controls.json',controls)
    def verify():
        assert pin(BASE/'prepared.json')==controls['prepared'] and pin(Path(__file__))==controls['runner']
        for name,wanted in spec['files'].items():assert pin(ROOT/name)==wanted,name
    own=psutil.Process();prior=own.cpu_affinity();own.cpu_affinity([0])
    state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[]);save(BASE/'processes.json',state)
    try:
        for mode in ['inputs','run']:
            verify();folder=BASE/'process'/mode;folder.mkdir(parents=True)
            waited=time.monotonic()
            with (folder/'preflight.jsonl').open('x') as log:
                while True:
                    sample=dict(seconds=time.monotonic()-waited,available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
                    log.write(json.dumps(sample)+'\n');log.flush()
                    assert sample['disk']>=limits['disk'] and sample['seconds']<limits['preflight_wait_seconds']
                    if sample['available']>=limits['preflight']:break
                    time.sleep(10)
            row=dict(mode=mode,complete=False,code=None,preflight=sample,samples=0,peak_rss=0);state['runs'].append(row);save(BASE/'processes.json',state)
            command=['dotnet',str(BASE/'bin/NaturalMeetings.dll'),str(ROOT),str(BASE),str(BASE/('output-'+mode)),mode]
            child=None;identity=None;started=time.monotonic()
            try:
                with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
                    own.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=ROOT,env=clean_env(),stdout=out,stderr=err,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                    finally:own.cpu_affinity([0])
                    process=psutil.Process(child.pid);identity=dict(pid=child.pid,birth=process.create_time());row['worker']=identity;save(BASE/'processes.json',state)
                    while child.poll() is None:
                        try:
                            assert process.create_time()==identity['birth'] and not process.children(recursive=True)
                            sample=dict(seconds=time.monotonic()-started,rss=process.memory_info().rss,available=psutil.virtual_memory().available,
                                disk=shutil.disk_usage(BASE).free,affinity=process.cpu_affinity(),**identity)
                        except psutil.NoSuchProcess:continue
                        samples.write(json.dumps(sample)+'\n');samples.flush();row['samples']+=1;row['peak_rss']=max(row['peak_rss'],sample['rss'])
                        assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss'] and sample['available']>=limits['available']
                        assert sample['disk']>=limits['disk'] and sample['affinity']==[2]
                        time.sleep(.5)
                    row['code']=child.wait();assert row['code']==0,(mode,row['code'])
            except BaseException:
                if child is not None and child.poll() is None and identity is not None and not absent(identity):
                    psutil.Process(identity['pid']).kill();child.wait(timeout=10)
                raise
            finally:
                row.update(complete=True,seconds=time.monotonic()-started)
                if child is not None:row['code']=child.poll()
                save(BASE/'processes.json',state)
            assert absent(identity);print(json.dumps(row),flush=True)
        verify();state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'processes.json',state);own.cpu_affinity(prior)


if __name__=='__main__':main()
