"""Digest-verified phase deployment, exact collection, and PID-plus-birth polling."""
from pathlib import Path
import argparse,json,subprocess,tarfile
from audit import pin,read,write

HOST='vermorel@74.178.91.76';KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
REMOTE='/home/vermorel/Onnx/artifacts/e5-fingerprint-balanced-20260920'
PSUTIL='/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'

def ssh(script):
    return subprocess.check_output(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8')

def launch(base,phase):
    assert not (base/('deployment-'+phase+'.json')).exists()
    if phase=='aa':
        ssh("from pathlib import Path\nassert not Path(%r).exists() and not Path(%r).exists()" % (REMOTE,REMOTE+'.tar.gz'))
        subprocess.run(['scp','-i',KEY,str(base/'payload.tar.gz'),HOST+':'+REMOTE+'.tar.gz'],check=True)
    else:
        gate=read(base/'aa-gate.json');assert gate['passed'] and gate['timing_passed'] and gate['frozen']==pin(base/'payload/frozen.json')
        ssh("from pathlib import Path\nassert not Path(%r).exists()" % (REMOTE+'/aa-gate.json'))
        subprocess.run(['scp','-i',KEY,str(base/'aa-gate.json'),HOST+':'+REMOTE+'/aa-gate.json'],check=True)
    script='base=Path(%r)\nphase=%r\narchive_pin=%r\ngate_hash=%r\n' % (REMOTE,phase,pin(base/'payload.tar.gz'),pin(base/'aa-gate.json')['sha256'] if phase=='compare' else None)
    script='from pathlib import Path\nimport hashlib,json,tarfile,os,sys,time,subprocess\nsys.path.insert(0,%r)\nimport psutil\n' % PSUTIL+script+r'''
def pin(p):
 with p.open('rb') as stream:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
if phase=='aa':
 archive=base.with_name(base.name+'.tar.gz');assert pin(archive)==archive_pin and not base.exists()
 with tarfile.open(archive) as tar:
  members=tar.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
  assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
  tar.extractall(base,filter='data')
meta=json.loads((base/'frozen.json').read_text())
for name,wanted in meta['files'].items():assert pin(base/name)==wanted,name
assert not (base/('deployment-'+phase+'.json')).exists() and not (base/('result-'+phase)).exists()
assert psutil.virtual_memory().available>=8*1024**3
assert os.statvfs(base).f_bavail*os.statvfs(base).f_frsize>=128*1024**2
if phase=='compare':
 assert pin(base/'aa-gate.json')['sha256']==gate_hash
 gate=json.loads((base/'aa-gate.json').read_text());assert gate['passed'] and gate['timing_passed']
 for item in gate['births']:
  try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
  except psutil.NoSuchProcess:pass
env=os.environ.copy();env['PYTHONPATH']=sys.path[0]
command=['python3','-B',str(base/'run.py'),'--payload',str(base),'--phase',phase]
if gate_hash:command+=['--gate-sha256',gate_hash]
with (base/('launcher-'+phase+'.stdout')).open('x') as stdout,(base/('launcher-'+phase+'.stderr')).open('x') as stderr:
 child=subprocess.Popen(command,cwd=base,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 birth=psutil.Process(child.pid).create_time()
value=dict(pid=child.pid,birth=birth,started=time.time(),command=command,frozen=pin(base/'frozen.json'),gate_sha256=gate_hash)
with (base/('deployment-'+phase+'.json')).open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(value))
'''
    value=json.loads(ssh(script));assert value['frozen']==pin(base/'payload/frozen.json');write(base/('deployment-'+phase+'.json'),value);print(value)

def poll(base,phase):
    script='from pathlib import Path\nimport sys,json,time\nsys.path.insert(0,%r)\nimport psutil\nb=Path(%r)\nphase=%r\n' % (PSUTIL,REMOTE,phase)+r'''
state=json.loads((b/('result-'+phase)/'identity.json').read_text());items=[state['supervisor']]
if state['runs'] and 'child' in state['runs'][-1]:items.append(state['runs'][-1]['child'])
observed=[]
for item in items:
 try:
  p=psutil.Process(item['pid']);observed.append(dict(expected=item,actual_birth=p.create_time(),status=p.status(),same_birth=p.create_time()==item['birth']))
 except psutil.NoSuchProcess:observed.append(dict(expected=item,absent=True))
print(json.dumps(dict(complete=state['complete'],code=state.get('code'),error=state.get('error'),elapsed=time.time()-state['started'],
 completed=sum(r.get('code')==0 for r in state['runs']),current=state['runs'][-1]['job'] if state['runs'] else None,observed=observed)))
if state['runs']:
 folder=b/('result-'+phase)/state['runs'][-1]['job']['name']
 for name in ['stdout.txt','stderr.txt']:
  if (folder/name).exists():print(name,(folder/name).read_text()[-700:])
'''
    print(ssh(script))

def collect(base,phase):
    assert not (base/('collected-'+phase)).exists()
    script='from pathlib import Path\nimport sys,json,hashlib,tarfile,time\nsys.path.insert(0,%r)\nimport psutil\nbase=Path(%r)\nphase=%r\n' % (PSUTIL,REMOTE,phase)+r'''
def pin(p):
 with p.open('rb') as stream:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
state=json.loads((base/('result-'+phase)/'identity.json').read_text());deployment=json.loads((base/('deployment-'+phase+'.json')).read_text())
assert state['supervisor']=={k:deployment[k] for k in ['pid','birth']}
births={deployment['pid']:deployment['birth']}
for run in state['runs']:births.update({int(pid):birth for pid,birth in run['members'].items()})
for pid,birth in births.items():
 try:assert psutil.Process(pid).create_time()!=birth,('Owned process live',pid,birth)
 except psutil.NoSuchProcess:pass
meta=json.loads((base/'frozen.json').read_text())
for name,wanted in meta['files'].items():assert pin(base/name)==wanted,name
assert pin(Path(meta['model']['path']))=={k:meta['model'][k] for k in ['bytes','sha256']}
receipt=base/('collection-'+phase+'.json');archive=base.with_name(base.name+'-'+phase+'-results.tar.gz')
assert not receipt.exists() and not archive.exists()
files={}
for path in sorted(base.rglob('*')):
 assert not path.is_symlink(),path
 if path.is_file():files[path.relative_to(base).as_posix()]=pin(path)
value=dict(phase=phase,created=time.time(),all_owned_processes_terminal=True,births=[dict(pid=p,birth=b) for p,b in sorted(births.items())],frozen=pin(base/'frozen.json'),files=files)
with receipt.open('x') as stream:json.dump(value,stream,indent=2)
with tarfile.open(archive,'x:gz') as stream:
 for name in list(files)+[receipt.name]:stream.add(base/name,arcname=name,recursive=False)
print(json.dumps(dict(collection=value,receipt=pin(receipt),archive=pin(archive))))
'''
    value=json.loads(ssh(script));assert value['collection']['frozen']==pin(base/'payload/frozen.json')
    archive=base/(phase+'-results.tar.gz');assert not archive.exists()
    subprocess.run(['scp','-i',KEY,HOST+':'+REMOTE+'-'+phase+'-results.tar.gz',str(archive)],check=True);assert pin(archive)==value['archive']
    expected=value['collection']['files']|{'collection-'+phase+'.json':value['receipt']}
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=[m.name for m in members];assert len(names)==len(set(names)) and set(names)==set(expected)
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        tar.extractall(base/('collected-'+phase),filter='data')
    for name,wanted in expected.items():assert pin(base/('collected-'+phase)/name)==wanted,name
    write(base/('collection-check-'+phase+'.json'),dict(passed=True,remote=value));print('Collected',phase,len(expected),'files; actual process births terminal.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['launch','poll','collect']);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True);a=p.parse_args()
    globals()[a.action](a.artifact.resolve(),a.phase)
