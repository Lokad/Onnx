"""Copy the exact sampled ORT ELF during an idle boundary of the active graph run."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
SAMPLES=ROOT/'artifacts/parakeet-ort-native-samples-20260924'
GRAPHS=ROOT/'artifacts/parakeet-pad-current-graphs-v2-amd-20260926'
OUT=ROOT/'artifacts/parakeet-ort-activation-review-20260926'
REMOTE='/dev/shm/lokad-parakeet-pad-current-graphs-v2-20260926'
SITE='/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
SSH=['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes',
     '-o','ConnectTimeout=20','vermorel@74.178.91.76']


def read(path):return json.loads(path.read_text(encoding='utf8'))
def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def save(path,value):
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    assert not (OUT/'binary-receipt.json').exists() and not (OUT/'failed.json').exists()
    closed=read(SAMPLES/'closed.json');assert closed['passed'] and closed['analysis']==pin(SAMPLES/'analysis.json')
    assert closed['inspection']==pin(SAMPLES/'native-inspection.json')
    inspection=read(SAMPLES/'native-inspection.json');analysis=read(SAMPLES/'analysis.json')
    assert inspection['binary']==analysis['loaded_binary']
    owner=read(GRAPHS/'deployment.json');assert owner==dict(pid=1126961,birth=1790456331.32)
    expected=inspection['binary'];assert expected['bytes']==30112440
    spec=dict(owner=owner,graph_payload=pin(GRAPHS/'payload.json'),sample_closure=pin(SAMPLES/'closed.json'),
              binary=expected,path=inspection['path'],helper=pin(Path(__file__)),inference_calls=0)
    OUT.mkdir(exist_ok=True)
    if (OUT/'prospective.json').exists():assert read(OUT/'prospective.json')==spec
    else:save(OUT/'prospective.json',spec)
    script=f'''
import os,sys,json,time,signal,hashlib
from pathlib import Path
sys.path.insert(0,{SITE!r})
import psutil
os.sched_setaffinity(0,{{0}})
base=Path({REMOTE!r});expected={owner!r}
def read(p):return json.loads(p.read_text())
def live(i):
 try:
  p=psutil.Process(i['pid']);return p.create_time()==i['birth'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
def expired(signum,frame):raise TimeoutError('Bounded binary transfer expired')
signal.signal(signal.SIGALRM,expired);signal.alarm(45)
assert live(expected)
state=read(base/'identity.json');assert state['supervisor']==expected and not state['complete']
if not state['runs'] or not all(r['complete'] and r['code']==0 for r in state['runs']):
 print(json.dumps(dict(ready=False,owner=expected,latest=None if not state['runs'] else state['runs'][-1]['name'])),file=sys.stderr)
 sys.exit(75)
process=psutil.Process(expected['pid']);paused=False
try:
 process.suspend();paused=True;started=time.monotonic()
 while process.status()!=psutil.STATUS_STOPPED and time.monotonic()-started<5:time.sleep(.01)
 assert live(expected) and process.status()==psutil.STATUS_STOPPED
 if process.children(recursive=True) or read(base/'identity.json')!=state:
  print(json.dumps(dict(ready=False,owner=expected,reason='boundary-race')),file=sys.stderr)
  sys.exit(75)
 ancestors={{os.getpid(),*[p.pid for p in psutil.Process().parents()]}}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid in ancestors or p.pid==expected['pid']:continue
  assert p.info['name'] not in ['dotnet','perf']
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
 with (base/'payload.json').open('rb') as stream:
  assert hashlib.file_digest(stream,'sha256').hexdigest()=={spec['graph_payload']['sha256']!r}
 source=Path({inspection['path']!r});before=source.stat()
 assert before.st_size=={expected['bytes']}
 digest=hashlib.sha256();total=0
 with source.open('rb') as stream:
  while True:
   data=stream.read(65536)
   if not data:break
   digest.update(data);total+=len(data);sys.stdout.buffer.write(data)
 sys.stdout.buffer.flush()
 assert total=={expected['bytes']} and digest.hexdigest()=={expected['sha256']!r}
 after=source.stat()
 assert (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)==(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
 assert read(base/'identity.json')==state
 assert not process.children(recursive=True)
 receipt=dict(passed=True,owner=expected,paused_seconds=time.monotonic()-started,
  completed_jobs=len(state['runs']),last_job=state['runs'][-1]['name'],
  binary=dict(bytes=total,sha256=digest.hexdigest()),no_inference_overlap=True,
  source_unchanged=True,inference_calls=0)
finally:
 if paused and live(expected) and process.status()==psutil.STATUS_STOPPED:process.resume()
 signal.alarm(0)
assert live(expected) and process.status()!=psutil.STATUS_STOPPED
receipt['same_owner_resumed']=True
print(json.dumps(receipt),file=sys.stderr)
'''
    compile(script,'bounded-ort-binary-transfer','exec')
    try:
        result=subprocess.run(SSH+['python3','-B','-'],input=script.encode(),capture_output=True,
                              timeout=70,creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired as error:
        if error.stdout:(OUT/'partial-binary.bin').write_bytes(error.stdout)
        save(OUT/'failed.json',dict(error='transport timeout; inspect the same graph owner',
                                 stderr=(error.stderr or b'').decode(errors='replace')))
        raise
    if result.returncode==75:
        assert not result.stdout;value=json.loads(result.stderr)
        with (OUT/'boundary-observations.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value));return
    if result.returncode:
        if result.stdout:(OUT/'partial-binary.bin').write_bytes(result.stdout)
        save(OUT/'failed.json',dict(code=result.returncode,stderr=result.stderr.decode(errors='replace')))
        raise AssertionError('Preserve failed binary inspection; do not retry unchanged')
    receipt=json.loads(result.stderr)
    assert receipt['passed'] and receipt['no_inference_overlap'] and receipt['same_owner_resumed']
    assert receipt['binary']==expected==dict(bytes=len(result.stdout),sha256=hashlib.sha256(result.stdout).hexdigest())
    target=OUT/'onnxruntime_pybind11_state.so'
    with target.open('xb') as stream:stream.write(result.stdout)
    save(OUT/'binary-receipt.json',dict(**receipt,prospective=pin(OUT/'prospective.json'),local=pin(target)))
    print(json.dumps(receipt))


if __name__=='__main__':main()
