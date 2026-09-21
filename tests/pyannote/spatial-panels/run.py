"""Four fresh local current/candidate workers, including complete application costs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from prepare import ROOT, BASE, TOOLS, INPUT, pin, save, clean_env
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def absent(identity):
    try:return psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess:return True


def main():
    assert not (BASE/'processes.json').exists()
    spec=json.loads((BASE/'manifest.json').read_text())
    assert spec['jobs']==['baseline','candidate','candidate','baseline']
    controls=dict(manifest=pin(BASE/'manifest.json'),runner=pin(Path(__file__)),
        jobs=spec['jobs'],limits=dict(seconds=1800,rss=8*1024**3,available=1024**3,
        preflight_available=10*1024**3,disk=20*1024**3,preflight_wait_seconds=3600))
    assert not (BASE/'run-controls.json').exists();save(BASE/'run-controls.json',controls)
    def verify():
        assert pin(BASE/'manifest.json')==controls['manifest'] and pin(Path(__file__))==controls['runner']
        for name,wanted in spec['files'].items():assert pin(ROOT/name)==wanted,name
    own=psutil.Process();prior=own.cpu_affinity();own.cpu_affinity([0])
    state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    save(BASE/'processes.json',state)
    try:
        for index,role in enumerate(spec['jobs']):
            verify();folder=BASE/'process'/f'{index}-{role}';folder.mkdir(parents=True)
            waited=time.monotonic()
            with (folder/'preflight.jsonl').open('x') as log:
                while True:
                    preflight=dict(seconds=time.monotonic()-waited,available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
                    log.write(json.dumps(preflight)+'\n');log.flush()
                    assert preflight['disk']>=20*1024**3
                    if preflight['available']>=10*1024**3:break
                    assert preflight['seconds']<3600,preflight
                    time.sleep(10)
            row=dict(index=index,role=role,complete=False,code=None,preflight=preflight,samples=0,peak_rss=0)
            state['runs'].append(row);save(BASE/'processes.json',state)
            command=['dotnet',str(BASE/'runtimes'/role/'Profile.dll'),str(ROOT),str(INPUT),str(BASE/'outputs'/f'{index}-{role}'),spec['cores'][role]['sha256']]
            child=None;identity=None;started=time.monotonic()
            try:
                with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
                    own.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=ROOT,env=clean_env(),stdout=out,stderr=err,
                        creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                    finally:own.cpu_affinity([0])
                    process=psutil.Process(child.pid);identity=dict(pid=child.pid,birth=process.create_time());row['worker']=identity
                    save(BASE/'processes.json',state)
                    while child.poll() is None:
                        try:
                            assert process.create_time()==identity['birth'] and not process.children(recursive=True)
                            sample=dict(seconds=time.monotonic()-started,rss=process.memory_info().rss,available=psutil.virtual_memory().available,
                                disk=shutil.disk_usage(BASE).free,affinity=process.cpu_affinity(),**identity)
                        except psutil.NoSuchProcess:continue
                        samples.write(json.dumps(sample)+'\n');samples.flush();row['samples']+=1;row['peak_rss']=max(row['peak_rss'],sample['rss'])
                        assert sample['seconds']<1800 and sample['rss']<8*1024**3 and sample['available']>=1024**3
                        assert sample['disk']>=20*1024**3 and sample['affinity']==[2]
                        time.sleep(.25)
                    row['code']=child.wait();assert row['code']==0,(index,role,row['code'])
            except BaseException:
                if child is not None and child.poll() is None and identity is not None and not absent(identity):
                    process=psutil.Process(identity['pid']);process.kill();child.wait(timeout=10)
                raise
            finally:
                row.update(complete=True,seconds=time.monotonic()-started)
                if child is not None:row['code']=child.poll()
                save(BASE/'processes.json',state)
            assert absent(identity);print(json.dumps(row),flush=True)
        verify();state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:
        state['complete']=True;save(BASE/'processes.json',state);own.cpu_affinity(prior)


if __name__=='__main__':main()
