"""Qualify full Parakeet trajectories with the isolated inclusive packing core."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-packing-qualification-20260921'
ADMISSION = ROOT/'artifacts/parakeet-packing-admission-completion-v2-20260921'
TRACE = ROOT/'artifacts/parakeet-performance-profile-v2-20260921'
PRODUCT = ROOT/'artifacts/parakeet-packing-admission-v2-20260921/candidate-source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
MANIFEST = ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/parakeet.json'
sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
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


def main():
    admission=read(ADMISSION/'prepared.json');assert admission['passed'];verify(admission['files'])
    preparation=read(ADMISSION/'processes.json');assert preparation['complete'] and preparation['code']==0
    terminal(preparation['supervisor'])
    for row in preparation['runs']:
        for pid,birth in row['members'].items():terminal(dict(pid=int(pid),birth=birth))
    assert pin(TRACE/'closed.json')['sha256']=='6d8ce878f99acf291cb20348bba948bd1de7f128298c774cf94adf84e855700d'
    closed=read(TRACE/'closed.json');assert closed['passed'];verify(closed['files'])
    BASE.mkdir();source=BASE/'source';source.mkdir();(BASE/'logs').mkdir()
    origin=ROOT/'tests/parakeet/performance-profile'
    program=(origin/'Program.cs').read_text(encoding='utf8')
    replacements={
        'd1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4':pin(PRODUCT/'Lokad.Onnx.dll')['sha256'],
        'e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb':pin(PRODUCT/'Lokad.Onnx.Data.dll')['sha256'],
        'for (int pass = 0; pass < 2; pass++)':'for (int pass = 0; pass < 1; pass++)',
        'files.Count == 2480 && held.Count == 9760 && traced.Count == 40':'files.Count == 1240 && held.Count == 4880 && traced.Count == 20',
        'Local full-corpus graph attribution and separate public controls; no matched native timing or new tensor-native verdict':
        'Local inclusive packing correctness, fixed 256 MiB encoder cap; no speed claim or new tensor-native verdict',
    }
    for old,new in replacements.items():assert program.count(old)==1,old;program=program.replace(old,new)
    needle='    object generation = Private(model, "generation");'
    census='''    foreach (var graph in graphs.Values) graph.CreateExecution(ExecutionOptions.Memory).Reset();
    Write("packing.json", graphs.ToDictionary(p => p.Key, p => new {
        maximum_packed_bytes = p.Value.MaximumPackedWeightBytes,
        retained_packed_bytes = p.Value.RetainedPackedWeightBytes,
        weights = p.Value.Initializers.Where(v => v.Key.StartsWith("packed:", StringComparison.Ordinal))
            .Select(v => new { name = v.Key, shape = v.Value.Dims, bytes = v.Value.Length * 4 }).ToArray()
    }));
'''
    assert program.count(needle)==1;program=program.replace(needle,census+needle)
    (source/'Program.cs').write_text(program,encoding='utf8')
    shutil.copy2(origin/'Profile.csproj',source/'Profile.csproj');shutil.copy2(ROOT/'tests/Shared/NpySupport.cs',source/'NpySupport.cs')
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    flags=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:UseSharedCompilation=false','-nr:false']
    commands=[['dotnet','restore','Profile.csproj',*flags,'--source',str(ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'),'--packages',str(BASE/'packages'),'-p:NuGetAudit=false'],
              ['dotnet','build','Profile.csproj','-c','Release',*flags,'--no-restore','--disable-build-servers','-p:FrozenProductDirectory='+str(PRODUCT)]]
    builds=[]
    for label,command in zip(('restore','build'),commands,strict=True):
        with (BASE/'logs'/(label+'.log')).open('x') as log:code=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        builds.append(dict(label=label,command=command,code=code));save(BASE/'builds.json',builds);assert code==0,label
    shutil.copytree(source/'bin/Release/net10.0',BASE/'bin')
    for p in PRODUCT.glob('*.dll'):shutil.copy2(p,BASE/'bin'/p.name)
    files={}
    for folder in (TOOLS,source,BASE/'bin'):
        for p in folder.iterdir():
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in (ADMISSION/'prepared.json',TRACE/'closed.json',origin/'Program.cs',origin/'Profile.csproj',MANIFEST):files[p.relative_to(ROOT).as_posix()]=pin(p)
    spec=read(MANIFEST)
    for entry in [*spec['models'].values(),spec['reference'],*[c['pcm'] for c in spec['cases']]]:
        assert pin(ROOT/entry['path'])=={k:entry[k] for k in ('bytes','sha256')}
        files[entry['path']]=pin(ROOT/entry['path'])
    save(BASE/'prepared.json',dict(passed=True,files=files,encoder_budget=256*1024**2,trace_calls=1240,public_calls=20,
         trace_reference=pin(TRACE/'trace-output/result.json'),scope='Correctness qualification; no timing comparison'))
    own=psutil.Process();previous=own.cpu_affinity();own.cpu_affinity([0])
    controller=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    try:
        for mode in ('trace','public'):
            child=None;state=dict(mode=mode,complete=False,code=None,samples=0,peak_rss=0,preflight_observations=[])
            controller['runs'].append(state);beginning=time.monotonic();output=BASE/(mode+'-output')
            try:
                while True:
                    row=dict(seconds=time.monotonic()-beginning,available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
                    state['preflight_observations'].append(row);save(BASE/'processes.json',controller)
                    assert row['seconds']<900 and row['disk']>=20*1024**3
                    if row['available']>=14*1024**3:break
                    time.sleep(15)
                state['preflight']=row
                command=['dotnet',str(BASE/'bin/Profile.dll'),str(ROOT),str(MANIFEST),str(output),mode]
                with (BASE/'logs'/(mode+'.log')).open('x') as log,(BASE/'logs'/(mode+'.samples.jsonl')).open('x') as samples:
                    own.cpu_affinity([2])
                    try:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                    finally:own.cpu_affinity([0])
                    worker=psutil.Process(child.pid);state['worker']=dict(pid=worker.pid,birth=worker.create_time());start=time.monotonic()
                    while child.poll() is None:
                        try:
                            assert worker.create_time()==state['worker']['birth'] and not worker.children(recursive=True)
                            row=dict(seconds=time.monotonic()-start,rss=worker.memory_info().rss,available=psutil.virtual_memory().available,
                                     disk=shutil.disk_usage(BASE).free,affinity=worker.cpu_affinity())
                        except psutil.NoSuchProcess:
                            if child.poll() is not None:break
                            raise
                        samples.write(json.dumps(row)+'\n');samples.flush();state['samples']+=1;state['peak_rss']=max(state['peak_rss'],row['rss']);save(BASE/'processes.json',controller)
                        assert row['seconds']<1200 and row['rss']<12*1024**3 and row['available']>=1024**3 and row['disk']>=20*1024**3 and row['affinity']==[2]
                        time.sleep(.25)
                    state['code']=child.wait();assert state['code']==0,state['code']
                assert read(output/'result.json')['passed'];verify(files);state['passed']=True
            except BaseException:
                state.update(error=traceback.format_exc())
                if child is not None and child.poll() is None:
                    worker=psutil.Process(child.pid)
                    if worker.create_time()==state['worker']['birth']:worker.kill();child.wait(timeout=15)
                raise
            finally:
                state['complete']=True
                if child is not None:state['code']=child.poll()
                save(BASE/'processes.json',controller)
            print(mode,'completed',flush=True)
        controller['code']=0
    except BaseException:
        controller.update(code=1,error=traceback.format_exc());raise
    finally:
        controller['complete']=True;save(BASE/'processes.json',controller);own.cpu_affinity(previous)


if __name__=='__main__':main()
