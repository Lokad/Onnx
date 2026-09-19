"""Finite sequential qualification on CPU 2; preserves source/binary identities and all process samples."""
from pathlib import Path
import argparse,hashlib,json,os,platform,subprocess,time,traceback
import psutil

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--python',type=Path,default=Path('C:/Python313/python.exe'));a=p.parse_args()
    root=Path(__file__).resolve().parents[3];base=a.artifact.resolve();inputs=base/'inputs/inputs.json';models=root/'models/parakeet-tdt-0.6b-v3'
    source=base/'source';source.mkdir();paths=[]
    for folder in ('tests/parakeet/recording','src/Lokad.Onnx.Data','src/Lokad.Onnx.CLI'):
        paths.extend(p for p in (root/folder).iterdir() if p.suffix in ('.py','.cs','.csproj'))
    paths.extend(root/p for p in ('tests/Shared/NpySupport.cs','tests/parakeet/transcribe/assets.json','tests/parakeet/transcribe/generate_reference.py','.agent/m4-parakeet-recording-20260919.md','external/onnx-asr/src/onnx_asr/asr.py'))
    pins={}
    for path in paths:
        rel=path.relative_to(root);target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());pins[rel.as_posix()]=sha(path)
    binaries={}
    for folder in ('bin','cli-bin'):
        for path in (base/folder).iterdir():
            if path.is_file():binaries[path.relative_to(base).as_posix()]=sha(path)
    for name in ('Lokad.Onnx.dll','Lokad.Onnx.Data.dll'):assert binaries['bin/'+name]==binaries['cli-bin/'+name]
    frozen=dict(source=pins,binaries=binaries,inputs_sha256=sha(inputs),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    with (base/'frozen.json').open('x',encoding='utf-8') as f:json.dump(frozen,f,indent=2)
    identity=dict(supervisor=os.getpid(),host=platform.platform(),started=time.time(),complete=False,runs=[]);parent=psutil.Process();prior_affinity=parent.cpu_affinity();parent.cpu_affinity([0])
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    jobs=[('native',[str(a.python),'-X','utf8',str(root/'tests/parakeet/recording/native.py'),'--models',str(models),'--inputs',str(inputs),'--source',str(root/'external/onnx-asr/src/onnx_asr/asr.py'),'--output',str(base/'native')]),
        ('managed',['dotnet',str(base/'bin/ParakeetRecordingReplay.dll'),str(models),str(inputs),str(base/'managed')])]
    for name,extra in [('connected',[]),('token-limit',['--max-tokens=2'])]:
        jobs.append(('cli-'+name,['dotnet',str(base/'cli-bin/Lokad.Onnx.CLI.dll'),'transcribe',str(models),str(base/'inputs/connected.wav'),'--model-type=parakeet','--recording','--json']+extra))
    save=lambda:(base/'processes.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    def verify():
        for rel,value in pins.items():assert sha(root/rel)==value,rel
        for rel,value in binaries.items():assert sha(base/rel)==value,rel
        assert sha(inputs)==frozen['inputs_sha256']
    try:
        save()
        for name,command in jobs:
            verify();row=dict(name=name,command=command,started=time.time(),code=None,peak_rss=0,samples=0);start=time.monotonic()
            with (base/(name+'.stdout')).open('x',encoding='utf-8') as out,(base/(name+'.stderr')).open('x',encoding='utf-8') as err,(base/(name+'-samples.jsonl')).open('x',encoding='utf-8') as samples:
                parent.cpu_affinity([2])
                try:child=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                finally:parent.cpu_affinity([0])
                process=psutil.Process(child.pid);row.update(pid=child.pid,create_time=process.create_time());identity['runs'].append(row);save()
                try:
                    while child.poll() is None:
                        members=[]
                        for member in [process]+process.children(recursive=True):
                            try:
                                v=dict(pid=member.pid,create_time=member.create_time(),rss=member.memory_info().rss,affinity=member.cpu_affinity(),cpu_seconds=sum(member.cpu_times()[:2]));assert v['affinity']==[2];members.append(v)
                            except psutil.NoSuchProcess:pass
                        rss=sum(v['rss'] for v in members);row['peak_rss']=max(row['peak_rss'],rss);row['samples']+=1
                        samples.write(json.dumps(dict(seconds=time.monotonic()-start,members=members))+'\n');samples.flush()
                        assert rss<20*1024**3 and time.monotonic()-start<1800,'Resource guard'
                        time.sleep(.5)
                finally:
                    if child.poll() is None:
                        if process.is_running() and process.create_time()==row['create_time']:process.kill()
                    child.wait();row.update(code=child.returncode,seconds=time.monotonic()-start);save()
                assert child.returncode==0,(name,child.returncode)
            verify();print(name,'complete',row['seconds'],row['peak_rss'],flush=True)
        identity['complete']=True
    except BaseException:
        identity['error']=traceback.format_exc();raise
    finally:identity['ended']=time.time();save();parent.cpu_affinity(prior_affinity)

if __name__=='__main__':main()
