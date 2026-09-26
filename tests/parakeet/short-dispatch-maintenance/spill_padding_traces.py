"""Move completed traces off tmpfs, preserving paths and every original byte."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/pad-application-diagnostic-amd'))
from run import ssh, PRELUDE, BASE
from protocol import pin, save

OUT = ROOT/'artifacts/parakeet-padding-trace-spill-20260926'


def main():
    assert not OUT.exists()
    result = json.loads(ssh(PRELUDE+f'''
import hashlib,shutil,zipfile
from protocol import pin,read,save
from remote import live
expected=dict(pid=1111767,birth=1790442351.02)
assert psutil.boot_time()==1789634288.0 and live(expected)
assert pin(base/'payload.json')=={pin(BASE/'payload.json')!r}
state=read(base/'identity.json');spec=read(base/'payload.json')
assert not state['complete'] and state['supervisor']==expected
assert [r['name'] for r in state['runs']]==spec['jobs'][:10]
assert all(r['complete'] and r['code']==0 for r in state['runs'])
assert spec['jobs'][10]=='current-export' and not (base/'current-export').exists()
owner=psutil.Process(expected['pid']);assert owner.create_time()==expected['birth']
owner.suspend();paused=time.time()
try:
 deadline=time.monotonic()+5
 while owner.status()!=psutil.STATUS_STOPPED and time.monotonic()<deadline:time.sleep(.01)
 assert owner.status()==psutil.STATUS_STOPPED and not owner.children(recursive=True)
 assert read(base/'identity.json')==state
 assert not any(live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
 ancestors={{os.getpid(),*[p.pid for p in psutil.Process().parents()]}}
 for process in psutil.process_iter(['name','cmdline']):
  if process.pid in ancestors or process.pid==owner.pid:continue
  assert process.info['name'] not in ['dotnet','perf']
  assert not (process.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(process.info['cmdline'] or []))
 paths=set();manifests={{}}
 for folder in Path('/dev/shm').glob('lokad-*'):
  for name in ['payload.json','stage.json','spec.json']:
   path=folder/name
   if not path.is_file():continue
   value=read(path);manifests[str(path)]=pin(path)
   paths.update(str(folder/n) for n in value.get('files',{{}}));paths.update(value.get('external',{{}}))
   for link in value.get('links',{{}}).values():
    if isinstance(link,dict) and isinstance(link.get('source'),str):paths.add(link['source'])
 protected={{str(Path(n).resolve()) for n in paths}}
 package=Path('/home/vermorel/.nuget/packages/microsoft.ml.onnxruntime/1.23.2')
 archive=package/'microsoft.ml.onnxruntime.1.23.2.nupkg';archive_id=pin(archive)
 names=['runtimes/ios/native/onnxruntime.xcframework.zip',
        'runtimes/osx-x64/native/libonnxruntime.dylib',
        'runtimes/osx-arm64/native/libonnxruntime.dylib',
        'runtimes/android/native/onnxruntime.aar',
        'runtimes/linux-arm64/native/libonnxruntime.so',
        'runtimes/win-arm64/native/onnxruntime.dll',
        'runtimes/win-x64/native/onnxruntime.dll']
 retired={{}};reclaim=0
 with zipfile.ZipFile(archive) as packed:
  for name in names:
   path=package/name
   assert path.resolve()==path and path.is_relative_to(package) and not path.is_symlink()
   assert str(path) not in protected and path.stat().st_nlink==1
   identity=pin(path)
   with packed.open(name) as stream:
    assert packed.getinfo(name).file_size==identity['bytes']
    assert hashlib.file_digest(stream,'sha256').hexdigest()==identity['sha256']
   retired[str(path)]=identity;reclaim+=path.stat().st_blocks*512
 spill=Path('/home/vermorel/Onnx/artifacts/parakeet-padding-trace-spill-20260926')
 assert spill.resolve()==spill and spill.parent==Path('/home/vermorel/Onnx/artifacts') and not spill.exists()
 traces={{}}
 for role in ['current','candidate']:
  path=base/(role+'-capture/capture.nettrace')
  assert path.resolve()==path and not path.is_symlink() and path.stat().st_nlink==1 and str(path) not in protected
  traces[str(path)]=dict(identity=pin(path),target=str(spill/(role+'.nettrace')))
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free,disk=psutil.disk_usage(package).free)
 assert before['disk']+reclaim-sum(r['identity']['bytes'] for r in traces.values())>=32*1024**2
 prospective=dict(owner=expected,archive=dict(path=str(archive),identity=archive_id),retired=retired,traces=traces,
                  manifests=manifests,before=before,paused=paused)
 spill.mkdir();save(spill/'prospective.json',prospective);save(base/'trace-spill-prospective.json',prospective)
 for name in retired:Path(name).unlink()
 assert pin(archive)==archive_id
 for name,row in traces.items():
  source=Path(name);target=Path(row['target']);temporary=source.with_suffix('.nettrace.disk-link')
  assert not target.exists() and not temporary.exists() and pin(source)==row['identity']
  with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
  assert pin(target)==row['identity']
  os.symlink(target,temporary);temporary.replace(source)
  assert source.resolve()==target and pin(source)==row['identity']
 assert pin(archive)==archive_id and all(not Path(n).exists() for n in retired)
 assert read(base/'identity.json')==state and pin(base/'payload.json')=={pin(BASE/'payload.json')!r}
 result=dict(passed=True,prospective=prospective,trace_bytes=sum(r['identity']['bytes'] for r in traces.values()),
             retired_foreign_runtime_bytes=sum(r['bytes'] for r in retired.values()),
             after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free,disk=psutil.disk_usage(package).free),
             all_trace_bytes_retained=True,all_canonical_package_bytes_retained=True)
 save(spill/'closed.json',result);save(base/'trace-spill.json',result)
finally:
 if live(expected) and owner.status()==psutil.STATUS_STOPPED:owner.resume()
assert live(expected) and owner.status()!=psutil.STATUS_STOPPED
result.update(resumed=time.time(),resumed_same_owner=True)
print(json.dumps(result))
''',300))
    OUT.mkdir();save(OUT/'closed.json',dict(**result,helper=pin(__file__)))
    print(json.dumps({k:v for k,v in result.items() if k!='prospective'}))


if __name__=='__main__':main()
