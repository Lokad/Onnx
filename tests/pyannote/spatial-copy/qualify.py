"""After timing, run complete suites and unchanged native shared-model contracts."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from prepare import ROOT, BASE, pin, save, clean_env
from run import absent, psutil


def main():
    spec=json.loads((BASE/'manifest.json').read_text());runs=json.loads((BASE/'processes.json').read_text())
    assert runs['complete'] and runs['code']==0 and absent(runs['supervisor']) and all(absent(r['worker']) for r in runs['runs'])
    assert not (BASE/'qualification.json').exists()
    for name,expected in spec['files'].items():assert pin(ROOT/name)==expected,name
    controller=psutil.Process()
    state=dict(complete=False,code=None,supervisor=dict(pid=controller.pid,birth=controller.create_time()),steps=[],models=[]);save(BASE/'qualification.json',state)
    source=BASE/'candidate-source';flags=['-c','Release','--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false']
    def step(name,command):
        with (BASE/'logs'/(name+'.log')).open('x') as log:
            code=subprocess.run(command,cwd=source,env=clean_env(),stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
        state['steps'].append(dict(name=name,command=command,code=code));save(BASE/'qualification.json',state)
        assert code==0,name;print(name,'passed',flush=True)
    try:
        for name,path in [('cli','src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),('tensors','tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj')]:
            step(name+'-build',['dotnet','build',path,*flags])
        for name in ('Backend','Tensors'):
            step(name.lower()+'-full-tests',['dotnet','test',f'tests/Lokad.Onnx.{name}.Tests/Lokad.Onnx.{name}.Tests.csproj',*flags,'--no-build'])
        assert pin(source/'src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll')==spec['cores']['candidate'],'Rebuilt core differs'
        old=ROOT/'artifacts/e5-fingerprint-product-v2-20260920/payload/tests/e5/fingerprint-product/bin/Release/net10.0'
        target=BASE/'shared-runtime';shutil.copytree(old,target)
        for suffix in ('dll','pdb'):shutil.copy2(BASE/'runtimes/candidate'/('Lokad.Onnx.'+suffix),target/('Lokad.Onnx.'+suffix))
        assert pin(target/'Replay.dll')['sha256']=='a50d3e965cf480844559b1f5856e3de318b5c8a269742a9e6504afb9cee6c8c0'
        assert pin(target/'Replay.dll')==pin(old/'Replay.dll')
        own=psutil.Process();prior=own.cpu_affinity();own.cpu_affinity([0])
        try:
            for mode,reference in [('e5',ROOT/'artifacts/e5-randomized-processes-20260921/payload/inputs'),('shared',ROOT/'artifacts/shared-regression-20260918/reference')]:
                folder=BASE/'shared-process'/mode;folder.mkdir(parents=True)
                waited=time.monotonic()
                with (folder/'preflight.jsonl').open('x') as preflight:
                    while True:
                        sample=dict(seconds=time.monotonic()-waited,available=psutil.virtual_memory().available,disk=shutil.disk_usage(BASE).free)
                        preflight.write(json.dumps(sample)+'\n');preflight.flush()
                        assert sample['seconds']<3600 and sample['disk']>=20*1024**3
                        if sample['available']>=10*1024**3:break
                        time.sleep(10)
                env=clean_env();env['LOKAD_ONNX_FINGERPRINT_STRINGS']='0'
                command=['dotnet',str(target/'Replay.dll'),mode,str(ROOT),str(reference),str(BASE/'shared-output'/mode),spec['cores']['candidate']['sha256']]
                child=None;identity=None;started=time.monotonic()
                row=dict(mode=mode,complete=False,code=None);state['models'].append(row)
                try:
                    with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
                        own.cpu_affinity([2])
                        try:child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
                        finally:own.cpu_affinity([0])
                        process=psutil.Process(child.pid);identity=dict(pid=child.pid,birth=process.create_time());row['worker']=identity;save(BASE/'qualification.json',state)
                        while child.poll() is None:
                            try:
                                assert process.create_time()==identity['birth'] and not process.children(recursive=True)
                                sample=dict(seconds=time.monotonic()-started,rss=process.memory_info().rss,available=psutil.virtual_memory().available,
                                    disk=shutil.disk_usage(BASE).free,affinity=process.cpu_affinity())
                            except psutil.NoSuchProcess:continue
                            samples.write(json.dumps(sample)+'\n');samples.flush()
                            assert sample['seconds']<900 and sample['rss']<8*1024**3 and sample['available']>=1024**3 and sample['disk']>=20*1024**3 and sample['affinity']==[2]
                            time.sleep(.25)
                        row['code']=child.wait();assert row['code']==0,mode
                except BaseException:
                    if child is not None and child.poll() is None and identity is not None and not absent(identity):
                        psutil.Process(identity['pid']).kill();child.wait(timeout=10)
                    raise
                finally:
                    row['complete']=True;save(BASE/'qualification.json',state)
                assert absent(identity);print(mode,'model contracts passed',flush=True)
        finally:own.cpu_affinity(prior)
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'qualification.json',state)


if __name__=='__main__':main()
