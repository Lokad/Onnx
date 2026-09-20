"""Guarded deployment after primary e5 closure, exact polling and streamed collection."""
from pathlib import Path
import argparse,json,subprocess,tarfile
from common import REMOTE,pin,read,write,verify

ROOT=Path(__file__).resolve().parents[3]
HOST='vermorel@74.178.91.76';KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
PSUTIL='/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'

def ssh(script):return subprocess.check_output(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8')
def prefix():return 'from pathlib import Path\nimport sys,json,hashlib,time,os,subprocess,tarfile\nsys.path.insert(0,%r)\nimport psutil\nbase=Path(%r)\n' % (PSUTIL,REMOTE)

def launch(base):
    verify(base/'payload');frozen=read(base/'frozen.json');assert frozen['archive']==pin(base/'payload.tar.gz') and not (base/'deployment.json').exists()
    primary=ROOT/'artifacts/e5-fingerprint-deployment-20260920';closed=read(primary/'closed.json')
    for name,want in closed['files'].items():assert pin(primary/name)==want,name
    final=read(primary/'final-verification.json');assert final['passed'] is True and final['phases']==closed['phases']
    births=final['terminal']['births'];eligibility=dict(primary_closed=pin(primary/'closed.json'),phases=closed['phases'],births=births)
    script=prefix()+'births=%r\n'%births+'''
assert not base.exists() and not base.with_name(base.name+'.tar.gz').exists()
for item in births:
 try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
 except psutil.NoSuchProcess:pass
assert psutil.virtual_memory().available>=8*1024**3
disk=os.statvfs(base.parent);free=disk.f_bavail*disk.f_frsize
assert free>=512*1024**2,('Free disk below512MiB',free)
print(json.dumps(dict(available=psutil.virtual_memory().available,free_disk=free,checked_at=time.time())))
'''
    preflight=json.loads(ssh(script))
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(base/'payload.tar.gz'),HOST+':'+REMOTE+'.tar.gz'],check=True)
    script=prefix()+'expected=%r\neligibility=%r\npsutil_directory=%r\n'%(pin(base/'payload.tar.gz'),eligibility,PSUTIL)+'''
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
archive=base.with_name(base.name+'.tar.gz');assert pin(archive)==expected and not base.exists()
with tarfile.open(archive) as tar:
 members=tar.getmembers();names=[m.name for m in members]
 assert len(names)==len(set(names)) and all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
 tar.extractall(base,filter='data')
sys.path.insert(0,str(base/'tools'));from common import verify
verify(base)
for item in eligibility['births']:
 try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
 except psutil.NoSuchProcess:pass
with (base/'eligibility.json').open('x') as f:json.dump(eligibility,f,indent=2)
command=['python3','-B',str(base/'tools/run.py'),'--payload',str(base)]
env=os.environ.copy();env['PYTHONPATH']=psutil_directory
with (base/'launcher.stdout').open('x') as stdout,(base/'launcher.stderr').open('x') as stderr:
 child=subprocess.Popen(command,cwd=base,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 birth=psutil.Process(child.pid).create_time()
value=dict(pid=child.pid,birth=birth,started=time.time(),command=command,bundle=pin(base/'bundle.json'))
with (base/'deployment.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
'''
    deployment=json.loads(ssh(script));assert deployment['bundle']==pin(base/'payload/bundle.json')
    write(base/'deployment.json',deployment);write(base/'deployment-preflight.json',dict(eligibility=eligibility,preflight=preflight));print(deployment)

def poll(base):
    script=prefix()+'''
state=json.loads((base/'result/identity.json').read_text());items=[state['supervisor']]
if state['runs'] and 'child' in state['runs'][-1]:items.append(state['runs'][-1]['child'])
observed=[]
for item in items:
 try:
  process=psutil.Process(item['pid']);observed.append(dict(expected=item,actual_birth=process.create_time(),status=process.status(),same_birth=process.create_time()==item['birth']))
 except psutil.NoSuchProcess:observed.append(dict(expected=item,absent=True))
print(json.dumps(dict(complete=state['complete'],code=state.get('code'),error=state.get('error'),runs=[dict(phase=r['phase'],code=r.get('code')) for r in state['runs']],observed=observed)))
if state['runs']:
 for name in ['stdout.txt','stderr.txt']:
  path=base/'result'/state['runs'][-1]['phase']/name
  if path.exists():print(name,path.read_text()[-1200:])
'''
    print(ssh(script))

def collect(base):
    assert not (base/'collection-check.json').exists()
    script=prefix()+'''
sys.path.insert(0,str(base/'tools'));from common import verify,pin
verify(base);state=json.loads((base/'result/identity.json').read_text())
assert 'ended' in state and 'code' in state
births=[state['supervisor']]
for run in state['runs']:births.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
for item in births:
 try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
 except psutil.NoSuchProcess:pass
assert not (base/'collection.json').exists()
files={}
for path in sorted(base.rglob('*')):
 assert not path.is_symlink(),path
 if path.is_file():files[path.relative_to(base).as_posix()]=pin(path)
value=dict(all_owned_processes_terminal=True,births=births,files=files,bundle=pin(base/'bundle.json'))
with (base/'collection.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(dict(collection=value,receipt=pin(base/'collection.json'))))
'''
    remote=json.loads(ssh(script));assert remote['collection']['bundle']==pin(base/'payload/bundle.json');write(base/'collection-remote.json',remote)
    expected=remote['collection']['files']|{'collection.json':remote['receipt']}
    script=prefix()+'wanted=%r\n'%remote['receipt']+'''
sys.path.insert(0,str(base/'tools'));from common import pin
assert pin(base/'collection.json')==wanted
value=json.loads((base/'collection.json').read_text())
for name,want in value['files'].items():assert pin(base/name)==want,name
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in list(value['files'])+['collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    archive=base/'results.tar.gz'
    with archive.open('xb') as output:subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script.encode(),stdout=output,check=True)
    with tarfile.open(archive) as tar:
        members=tar.getmembers();names=[m.name for m in members]
        assert len(names)==len(set(names)) and set(names)==set(expected)
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        tar.extractall(base/'collected',filter='data')
    for name,want in expected.items():assert pin(base/'collected'/name)==want,name
    write(base/'collection-check.json',dict(passed=True,streamed=True,remote=remote,archive=pin(archive)));print('Collected',len(expected),'files; every original birth is terminal.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['launch','poll','collect']);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args()
    globals()[args.action](args.artifact.resolve())
