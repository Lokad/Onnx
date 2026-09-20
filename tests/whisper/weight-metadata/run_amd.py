"""Reproduce the exact decoder-only control diagnostic on AMD after failed-job closure."""
from pathlib import Path
import importlib.util,json,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/whisper/weight-sharing'))
from deploy import ssh,KEY,HOST
from protocol import pin,read,write
BASE=ROOT/'artifacts/whisper-weight-metadata-20260920'
REMOTE='/home/vermorel/Onnx/artifacts/whisper-weight-metadata-20260920'
OLD='/home/vermorel/Onnx/artifacts/whisper-weight-sharing-20260920'


def main():
    state=read(BASE/'identity.json');assert state['complete'] and state['code']==0
    frozen=read(BASE/'frozen.json')
    for name,wanted in frozen['files'].items():assert pin(BASE/name)==wanted,name
    source=ROOT/'artifacts/whisper-weight-sharing-20260920'
    closure=read(source/'failure-closed.json');assert closure['closure_passed'] and not closure['campaign_passed']
    for name,wanted in closure['files'].items():assert pin(ROOT/name)==wanted,name
    folder=BASE/'amd-v2';folder.mkdir()
    uploads={f'bin/{p.name}':p for p in (BASE/'bin').iterdir() if p.suffix in ['.dll','.json']}
    for name in ['Program.cs','WhisperWeightMetadata.csproj']:uploads['source/'+name]=BASE/'source'/name
    uploads['prospective-plan.md']=ROOT/'.agent/m4-whisper-weight-metadata-20260920.md'
    uploads['run_amd.py']=Path(__file__);uploads['prior-failure.json']=source/'failure-closed.json'
    old=read(source/'frozen.json');links={}
    for name,path in list(uploads.items()):
        if old['files'].get(name)==pin(path):links[name]=pin(path);del uploads[name]
    archive=folder/'payload.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in uploads.items():tar.add(path,arcname=name,recursive=False)
    remote_tar='/dev/shm/whisper-weight-metadata-20260920-payload-v2.tar.gz'
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(archive),HOST+':'+remote_tar],check=True)
    script='''from pathlib import Path
import os,sys,json,hashlib,tarfile,subprocess,time,traceback
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);prior=Path(%r);archive=Path(%r)
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
assert not base.exists();assert pin(archive)==%r
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
assert pin(prior/'collection.json')==%r
base.mkdir()
for name,wanted in %r.items():
 assert pin(prior/name)==wanted
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True);os.link(prior/name,p)
with tarfile.open(archive) as tar:
 for item in tar.getmembers():assert not (base/item.name).exists()
 tar.extractall(base,filter='data')
for name,wanted in %r.items():assert pin(base/name)==wanted,name
models=Path('/home/vermorel/Onnx/models/whisper-large-v3-turbo/onnx')
external={str(models/n):v for n,v in %r.items()}
runtime=Path('/home/vermorel/.dotnet/shared/Microsoft.NETCore.App/10.0.8')
external.update({str(p):pin(p) for p in sorted(runtime.rglob('*')) if p.is_file()})
external['/home/vermorel/.dotnet/dotnet']=pin(Path('/home/vermorel/.dotnet/dotnet'))
for name,wanted in external.items():assert pin(Path(name))==wanted,name
limits=dict(seconds=120,rss=8*1024**3,available=4*1024**3,preflight=8*1024**3,disk=32*1024**2,preflight_disk=64*1024**2)
files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
write(base/'frozen.json',dict(files=files,external=external,limits=limits))
parent=psutil.Process();parent.cpu_affinity([0]);state=dict(complete=False,code=None,supervisor=dict(pid=parent.pid,birth=parent.create_time()),runs=[],frozen=pin(base/'frozen.json'))
def save():
 p=base/'identity.tmp';p.write_text(json.dumps(state,indent=2));p.replace(base/'identity.json')
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))};save()
try:
 for mode in ['unshared','shared']:
  folder=base/mode;folder.mkdir();available=psutil.virtual_memory().available;disk=psutil.disk_usage(str(base)).free
  assert available>=limits['preflight'] and disk>=limits['preflight_disk']
  row=dict(mode=mode,complete=False,code=None,started=time.time(),preflight_available=available,preflight_disk=disk,samples=0,peak_rss=0)
  state['runs'].append(row);save();child=None;start=time.monotonic()
  try:
   with (folder/'stdout.txt').open('x') as out,(folder/'stderr.txt').open('x') as err,(folder/'samples.jsonl').open('x') as samples:
    parent.cpu_affinity([2])
    try:child=subprocess.Popen(['/home/vermorel/.dotnet/dotnet',str(base/'bin/WhisperWeightMetadata.dll'),str(models),mode,str(folder/'worker')],cwd=base,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL)
    finally:parent.cpu_affinity([0])
    process=psutil.Process(child.pid);row['child']=dict(pid=child.pid,birth=process.create_time());save()
    while child.poll() is None:
     try:
      assert process.create_time()==row['child']['birth']
      sample=dict(seconds=time.monotonic()-start,rss=process.memory_info().rss,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(base)).free,affinity=process.cpu_affinity())
     except psutil.NoSuchProcess:break
     samples.write(json.dumps(sample)+chr(10));samples.flush();row['samples']+=1;row['peak_rss']=max(row['peak_rss'],sample['rss']);save()
     assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss'] and sample['available']>=limits['available'] and sample['disk']>=limits['disk'] and sample['affinity']==[2]
     time.sleep(.5)
    row['code']=child.wait();assert row['code']==0
   value=json.loads((folder/'worker/result.json').read_text());assert value['passed'] and value['runtime']=='.NET 10.0.8' and value['affinity']==4 and value['processor_count']==1 and value['flags']=={}
   row['result']=pin(folder/'worker/result.json')
  except BaseException as error:
   row['error']=repr(error)
   if child is not None and child.poll() is None:
    if psutil.Process(child.pid).create_time()==row['child']['birth']:child.kill();child.wait(timeout=10)
   raise
  finally:row.update(complete=True,seconds=time.monotonic()-start,ended=time.time());save()
 for name,wanted in files.items():assert pin(base/name)==wanted,name
 for name,wanted in external.items():assert pin(Path(name))==wanted,name
 state['code']=0
except BaseException as error:state['error']=repr(error);traceback.print_exc()
finally:state['complete']=True;save()
result=Path('/dev/shm/whisper-weight-metadata-20260920-results.tar.gz');assert not result.exists()
collected={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
write(base/'collection.json',dict(files=collected,external_verified=len(external),code=state['code']))
with tarfile.open(result,'w:gz') as tar:
 for name in [*collected,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
print(json.dumps(dict(archive=pin(result),collection=pin(base/'collection.json'),state=state)))
'''%(REMOTE,OLD,remote_tar,pin(archive),closure['births'],pin(source/'collected/collection.json'),links,{n:pin(p) for n,p in uploads.items()},read(source/'weight-census.json')['models'])
    (folder/'remote-script.py').write_text(script,encoding='utf-8')
    compile(script,'frozen-remote-diagnostic','exec')
    result=json.loads(ssh(script));write(folder/'transfer.json',result)
    archive=folder/'results.tar.gz';subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':/dev/shm/whisper-weight-metadata-20260920-results.tar.gz',str(archive)],check=True)
    assert pin(archive)==result['archive'];collected=folder/'collected';collected.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(collected,filter='data')
    assert pin(collected/'collection.json')==result['collection']
    for name,wanted in read(collected/'collection.json')['files'].items():assert pin(collected/name)==wanted,name
    print(json.dumps(dict(code=result['state']['code'],runs=result['state']['runs'])))


if __name__=='__main__':main()
