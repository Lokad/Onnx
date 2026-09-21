"""Compare actual bounded encoder mappings, preserving the original core control."""
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
BASE=ROOT/'artifacts/parakeet-packing-residency-20260921'
ADMISSION=ROOT/'artifacts/parakeet-packing-admission-completion-v2-20260921'
QUALIFICATION=ROOT/'artifacts/parakeet-packing-qualification-v2-20260921'
BASELINE=ROOT/'artifacts/parakeet-packing-admission-20260921/baseline-source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'
PRODUCT=ROOT/'artifacts/parakeet-packing-admission-v2-20260921/candidate-source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
sys.path.insert(0,str(ROOT/'tests/parakeet/packing-qualification'))
from qualify import pin,read,save,verify,terminal,psutil


def main():
    admission=read(ADMISSION/'prepared.json');assert admission['passed'];verify(admission['files'])
    qualification=read(QUALIFICATION/'closed.json');assert qualification['passed'];verify(qualification['files'])
    state=read(QUALIFICATION/'processes.json');terminal(state['supervisor'])
    for row in state['runs']:terminal(row['worker'])
    spec=read(ROOT/'artifacts/audio-ort-baseline-v2-20260919/inputs/parakeet.json');files={}
    for value in spec['models'].values():
        assert pin(ROOT/value['path'])=={k:value[k] for k in ('bytes','sha256')};files[value['path']]=pin(ROOT/value['path'])
    BASE.mkdir();source=BASE/'source';source.mkdir();(BASE/'logs').mkdir()
    shutil.copy2(TOOLS/'Program.cs',source/'Program.cs')
    shutil.copy2(ROOT/'tests/parakeet/packing-census/Census.csproj',source/'Census.csproj')
    flags=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:UseSharedCompilation=false','-nr:false']
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    commands=[['dotnet','restore','Census.csproj',*flags,'--source',str(ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'),'--packages',str(BASE/'packages'),'-p:NuGetAudit=false'],
              ['dotnet','build','Census.csproj','-c','Release',*flags,'--no-restore','--disable-build-servers','-p:FrozenProductDirectory='+str(PRODUCT)]]
    for label,command in zip(('restore','build'),commands,strict=True):
        with (BASE/'logs'/(label+'.log')).open('x') as log:code=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        assert code==0,label
    for role,binaries in [('baseline',BASELINE),('candidate',PRODUCT)]:
        folder=BASE/role;shutil.copytree(source/'bin/Release/net10.0',folder)
        for p in binaries.glob('*.dll'):shutil.copy2(p,folder/p.name)
    for folder in (TOOLS,source,BASE/'baseline',BASE/'candidate'):
        for p in folder.iterdir():
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in (ADMISSION/'prepared.json',QUALIFICATION/'closed.json'):files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,files=files,budgets=[256*1024**2,512*1024**2,2032*1024**2]))
    own=psutil.Process();previous=own.cpu_affinity();own.cpu_affinity([0])
    state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    try:
        for role in ('baseline','candidate'):
            for mib in (256,512,2032):
                name=f'{role}-{mib}';folder=BASE/role;output=BASE/(name+'.json');child=None
                run=dict(name=name,complete=False,code=None,samples=0,peak_rss=0);state['runs'].append(run);save(BASE/'processes.json',state)
                assert psutil.virtual_memory().available>=10*1024**3 and shutil.disk_usage(BASE).free>=20*1024**3
                command=['dotnet',str(folder/'Census.dll'),str(ROOT/spec['models']['encoder-model.onnx']['path']),str(mib*1024**2),pin(folder/'Lokad.Onnx.dll')['sha256'],pin(folder/'Lokad.Onnx.Data.dll')['sha256'],str(output)]
                try:
                    with (BASE/'logs'/(name+'.log')).open('x') as log,(BASE/'logs'/(name+'.samples.jsonl')).open('x') as samples:
                        own.cpu_affinity([2])
                        try:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                        finally:own.cpu_affinity([0])
                        worker=psutil.Process(child.pid);run['worker']=dict(pid=worker.pid,birth=worker.create_time());start=time.monotonic()
                        while child.poll() is None:
                            try:
                                assert worker.create_time()==run['worker']['birth'] and not worker.children(recursive=True)
                                row=dict(seconds=time.monotonic()-start,rss=worker.memory_info().rss,available=psutil.virtual_memory().available,affinity=worker.cpu_affinity())
                            except psutil.NoSuchProcess:
                                if child.poll() is not None:break
                                raise
                            samples.write(json.dumps(row)+'\n');samples.flush();run['samples']+=1;run['peak_rss']=max(run['peak_rss'],row['rss']);save(BASE/'processes.json',state)
                            assert row['seconds']<180 and row['rss']<8*1024**3 and row['available']>=1024**3 and row['affinity']==[2]
                            time.sleep(.25)
                        run['code']=child.wait();assert run['code']==0
                    result=read(output);assert result['passed'];run.update(passed=True,retained_bytes=result['retained_bytes'],weights=len(result['weights']))
                except BaseException:
                    run['error']=traceback.format_exc()
                    if child is not None and child.poll() is None:
                        worker=psutil.Process(child.pid)
                        if worker.create_time()==run['worker']['birth']:worker.kill();child.wait(timeout=15)
                    raise
                finally:run['complete']=True;save(BASE/'processes.json',state)
                print(name,run['weights'],run['retained_bytes'],flush=True)
        verify(files);state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'processes.json',state);own.cpu_affinity(previous)


if __name__=='__main__':main()
