"""Supervise a fixed single-CPU audio comparison, preserving complete process evidence."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,platform,subprocess,sys,time,traceback
import psutil

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--mode',choices=['conformance','timing'],required=True);p.add_argument('--name',required=True);a=p.parse_args();root=a.root.resolve();base=a.artifact.resolve()
    assert psutil.__version__=='7.0.0'
    out=base/a.name;out.mkdir()
    files={p.relative_to(root).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cs','.csproj')}
    for path in [root/'tests/Shared/NpySupport.cs',root/'tests/pyannote/diarization/native_rules.py',root/'eng/campaign_processes.py']:
        files[path.relative_to(root).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
    for folder in ('inputs','bin'):
        for path in (base/folder).iterdir():
            if path.suffix in ('.dll','.json'):files[path.relative_to(root).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
    assert sha(base/'bin/Lokad.Onnx.dll')=='05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2'
    assert sha(base/'bin/Lokad.Onnx.Data.dll')=='27598aa8d8c6b97a1415302cf3aaced1adcf53b20b64734c0c0e047492ca069d'
    source=dict(files=files,product_source='8732831b52a97b009ab3edbd5319a56269e19449',mode=a.mode,plan_sha256=sha(root/'.agent/m4-audio-ort-baseline-20260919.md'))
    if a.mode=='timing':
        conformance=json.loads((base/'conformance/identity.json').read_text(encoding='utf-8'));assert conformance['complete'] and len(conformance['runs'])==4 and all(j['code']==0 for j in conformance['runs'])
        previous=json.loads((base/'conformance/frozen.json').read_text(encoding='utf-8'))
        assert previous['files']==files,'Conformance/measured payload differs'
        source['conformance_identity_sha256']=sha(base/'conformance/identity.json')
    with (out/'frozen.json').open('x',encoding='utf-8') as f:json.dump(source,f,indent=2)
    with (out/'prospective-plan.md').open('xb') as f:f.write((root/'.agent/m4-audio-ort-baseline-20260919.md').read_bytes())
    spec=importlib.util.spec_from_file_location('process_accounting',root/'eng/campaign_processes.py');account=importlib.util.module_from_spec(spec);spec.loader.exec_module(account)
    parent=psutil.Process();old_affinity=parent.cpu_affinity();parent.cpu_affinity([0]);identity=dict(supervisor=os.getpid(),started=time.time(),mode=a.mode,host=platform.platform(),supervisor_affinity=[0],runs=[],complete=False)
    (out/'cpu.txt').write_text(subprocess.check_output(['pwsh','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json'],text=True),encoding='utf-8')
    (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True),encoding='utf-8')
    save=lambda:(out/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    def verify():
        for rel,pin in files.items():assert sha(root/rel)==pin['sha256'] and (root/rel).stat().st_size==pin['bytes'],rel
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH=os.pathsep.join(str(root/path) for path in ('artifacts/pyannote-diarization-20260919/python','artifacts/pyannote-clustering-20260919/python')))
    schedule=[(family,engine) for family in ('parakeet','pyannote') for engine in (('ort','managed') if a.mode=='conformance' else ('managed','ort','ort','managed'))]
    try:
        verify();save()
        for index,(family,engine) in enumerate(schedule):
            name=f'{index:02d}-{family}-{engine}';destination=out/name;manifest=base/'inputs'/(family+'.json')
            command=['dotnet',str(base/'bin/AudioBenchmark.dll')] if engine=='managed' else ['C:/Python313/python.exe','-X','utf8',str(root/'tests/audio/comparison/native.py')]
            command.extend([str(root),str(manifest),str(destination),a.mode])
            row=dict(index=index,family=family,engine=engine,name=name,command=command,started=time.time(),samples=[],members={},peak_rss=0,code=None)
            before=account.snapshot();(out/(name+'-pre.json')).write_text(json.dumps(before),encoding='utf-8');start=time.monotonic()
            with (out/(name+'.log')).open('x',encoding='utf-8') as log:
                parent.cpu_affinity([2])
                try:child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                finally:parent.cpu_affinity([0])
                process=psutil.Process(child.pid);row.update(pid=child.pid,create_time=process.create_time());identity['runs'].append(row);save()
                try:
                    while child.poll() is None:
                        members=[]
                        for member in [process]+process.children(recursive=True):
                            try:
                                item=dict(pid=member.pid,create_time=member.create_time(),rss=member.memory_info().rss,affinity=member.cpu_affinity(),cpu_seconds=sum(member.cpu_times()[:2]),threads=member.num_threads())
                                assert item['affinity']==[2],item
                                members.append(item);row['members'][str(member.pid)]=dict(create_time=item['create_time'],affinity=item['affinity'])
                            except psutil.NoSuchProcess:pass
                        rss=sum(m['rss'] for m in members);row['peak_rss']=max(row['peak_rss'],rss)
                        row['seconds']=time.monotonic()-start;row['samples'].append(dict(seconds=row['seconds'],rss=rss,members=members));save()
                        assert rss<20*1024**3 and row['seconds']<3600,'Resource guard'
                        time.sleep(.5)
                finally:
                    if child.poll() is None:
                        for member in process.children(recursive=True):member.kill()
                        child.kill();child.wait()
                    row['code']=child.wait();row['ended']=time.time();row['seconds']=time.monotonic()-start;save()
            after=account.snapshot();(out/(name+'-post.json')).write_text(json.dumps(after),encoding='utf-8');row['accounting']=account.foreign_fraction(before,after,os.getpid());save()
            assert row['code']==0 and not psutil.pid_exists(row['pid']),row
            verify();print('Complete',name,row['seconds'],'RSS',row['peak_rss'],flush=True)
        identity['complete']=True;save()
    except BaseException:
        identity['error']=traceback.format_exc();save();raise
    finally:parent.cpu_affinity(old_affinity)

if __name__=='__main__':main()
