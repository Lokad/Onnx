import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-packing-budgets-20260921'
QUALIFIED=ROOT/'artifacts/parakeet-packing-qualification-v2-20260921'
ADMISSION=ROOT/'artifacts/parakeet-packing-admission-completion-v2-20260921'
PRODUCT=ROOT/'artifacts/parakeet-packing-admission-v2-20260921/candidate-source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
DATA_SOURCE=ROOT/'artifacts/parakeet-packing-admission-v2-20260921/candidate-source/src/Lokad.Onnx.Data'
FEED=ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
MANIFEST=ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/parakeet.json'
NATIVE=ROOT/'artifacts/parakeet-transcription-20260919/frozen'
NATIVE_BASELINE=ROOT/'artifacts/pyannote-optimized-parakeet-20260921'
REFERENCE=NATIVE/'reference/manifest.json'
SITE=ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
sys.path.insert(0,str(SITE))
import psutil

FLAGS=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:UseSharedCompilation=false','-nr:false']


def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf8'))
def verify(files):
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
def save(p,value):
    for attempt in range(20):
        try:
            t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)
def terminal(identity):
    try:assert psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess:pass
def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def worker(controller,state_path,name,command,cwd,expected,preflight_gib,rss_gib,seconds,allow_children,output):
    """Run one owned process tree with evidence written before each limit check."""
    own=psutil.Process();previous=own.cpu_affinity();own.cpu_affinity([0]);child=None
    row=dict(name=name,command=list(map(str,command)),complete=False,code=None,expected=expected,
             samples=0,peak_rss=0,members={},preflight_observations=[])
    controller['runs'].append(row);start=time.monotonic();save(state_path,controller)
    try:
        while True:
            sample=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
            row['preflight_observations'].append(sample);save(state_path,controller)
            assert sample['seconds']<900 and sample['disk']>=20*1024**3,'Preflight refusal'
            if sample['available']>=preflight_gib*1024**3:break
            time.sleep(15)
        row['preflight']=sample
        with (BASE/'logs'/(name+'.log')).open('x') as log,(BASE/'logs'/(name+'.samples.jsonl')).open('x') as samples:
            own.cpu_affinity([2])
            try:child=subprocess.Popen(row['command'],cwd=cwd,env=clean_env(),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
            finally:own.cpu_affinity([0])
            process=psutil.Process(child.pid);row['worker']=dict(pid=process.pid,birth=process.create_time());row['members'][str(process.pid)]=process.create_time()
            start=time.monotonic();last_scan=-5.;output_bytes=0
            while child.poll() is None:
                members=[]
                try:
                    assert process.create_time()==row['worker']['birth']
                    for p in [process]+process.children(recursive=True):
                        try:
                            birth=p.create_time();assert row['members'].get(str(p.pid),birth)==birth
                            row['members'][str(p.pid)]=birth
                            members.append(dict(pid=p.pid,birth=birth,name=p.name(),rss=p.memory_info().rss,affinity=p.cpu_affinity()))
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:
                    if child.poll() is not None:break
                    raise
                elapsed=time.monotonic()-start
                if output is not None and elapsed-last_scan>=5:
                    output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file());last_scan=elapsed
                sample=dict(seconds=elapsed,rss=sum(p['rss'] for p in members),available=psutil.virtual_memory().available,
                    disk=shutil.disk_usage(BASE).free,members=members,output_bytes=output_bytes)
                samples.write(json.dumps(sample)+'\n');samples.flush();row['samples']+=1;row['peak_rss']=max(row['peak_rss'],sample['rss']);save(state_path,controller)
                assert allow_children or len(members)<=1,sample
                assert elapsed<seconds and sample['rss']<rss_gib*1024**3 and sample['available']>=1024**3 and sample['disk']>=20*1024**3,sample
                assert all(p['affinity']==[2] for p in members) and output_bytes<=1024**3,sample
                time.sleep(.25)
            row['code']=child.wait();assert row['code'] in expected,(name,row['code'])
        return row
    except BaseException:
        row['error']=traceback.format_exc()
        for pid,birth in reversed(list(row['members'].items())):
            try:
                p=psutil.Process(int(pid))
                if p.create_time()==birth:p.kill()
            except psutil.NoSuchProcess:pass
        if child is not None:child.wait(timeout=15)
        raise
    finally:
        row.update(complete=True,seconds=time.monotonic()-start)
        if child is not None:row['code']=child.poll()
        save(state_path,controller);own.cpu_affinity(previous)
