"""Sample the original native application with perf, retaining exact instruction addresses."""
import ast
import base64
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from run import BASE as PRIOR, APP, ROOT, SSH, SITE, pin, read, write

BASE = ROOT/'artifacts/parakeet-ort-native-samples-20260924'
REMOTE = '/dev/shm/lokad-parakeet-ort-native-samples-20260924'


def main():
    assert not BASE.exists()
    prior = read(PRIOR/'closed.json'); assert prior['passed']
    assert prior['analysis'] == pin(PRIOR/'analysis.json')
    payload = read(APP/'payload.json')
    metadata = ROOT/'artifacts/parakeet-ort-native-capabilities-20260924'
    assert read(metadata/'closed.json')['passed']
    owners = prior['terminal_owners'][:]
    for folder in ['parakeet-ort-graphs-amd-v2-20260924','parakeet-ort-small-graphs-amd-20260924']:
        state = read(ROOT/'artifacts'/folder/'transfer.json')['state']
        assert state['complete']
        owners += [state['supervisor']]+[r['owner'] for r in state['runs']]
    BASE.mkdir()
    spec = dict(source=pin(__file__), original_profile=pin(PRIOR/'closed.json'),
        capabilities=pin(metadata/'closed.json'), app_payload=pin(APP/'payload.json'),
        native=pin(APP/'collected/runtime/native.py'), manifest=pin(APP/'collected/manifests/current-parakeet.json'),
        event='cpu-clock:u', frequency=199, stack_bytes=4096, clock='mono', requests=80,
        limits=dict(preflight_available=8*1024**3,preflight_tmpfs=2*1024**3,rss=6*1024**3,
                    available=1024**3,tmpfs=1024**3,output=512*1024**2,seconds=300))
    write(BASE/'prepared.json', spec)
    script = f'''import os,sys,json,time,subprocess,traceback,hashlib,shutil
from pathlib import Path
sys.path.insert(0,{SITE!r})
import psutil
os.sched_setaffinity(0,{{0}})
prior=Path('/dev/shm/lokad-parakeet-ort-diagnosis-20260924')
sys.path.insert(0,str(prior))
from remote import live,pin,read,save
assert all(not live(i) for i in {owners!r})
app=Path('/dev/shm/lokad-parakeet-prepared-recurrence-app-20260924')
assert pin(app/'payload.json')=={spec['app_payload']!r}
assert pin(app/'runtime/native.py')=={spec['native']!r}
assert pin(app/'manifests/current-parakeet.json')=={spec['manifest']!r}
base=Path({REMOTE!r});assert not base.exists();base.mkdir()
own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),members={{}},samples=0)
save(base/'state.json',state)
pre=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
save(base/'preflight.json',pre)
assert pre['available']>=8*1024**3 and pre['tmpfs']>=2*1024**3
env={{k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}}
env.pop('PYTHONOPTIMIZE',None)
values=dict(PYTHONPATH=os.pathsep.join({payload['python_paths']!r}),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',DEBUGINFOD_URLS='')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:values[k]='1'
env.update(values)
command=['sudo','-n','/usr/bin/perf','record','--no-buildid-cache','--clockid','mono','-e','cpu-clock:u','-F','199','--call-graph','dwarf,4096',
 '-o',str(base/'perf.data'),'--','sudo','-n','-u','vermorel','/usr/bin/taskset','-c','2','/usr/bin/env',
 *[k+'='+v for k,v in values.items()],sys.executable,'-B',str(app/'runtime/native.py'),str(app/'assets'),
 str(app/'manifests/current-parakeet.json'),str(base/'requests'),'timing']
state['command']=command;save(base/'state.json',state)
import importlib.util
module=importlib.util.spec_from_file_location('accounting',app/'runtime/campaign_processes.py')
account=importlib.util.module_from_spec(module);module.loader.exec_module(account)
before=account.snapshot();save(base/'cpu-before.json',before)
started=time.monotonic();child=None
try:
 with (base/'stdout').open('x') as stdout,(base/'stderr').open('x') as stderr,(base/'resources.jsonl').open('x') as log:
  child=subprocess.Popen(command,cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
  p=psutil.Process(child.pid);state['owner']=dict(pid=p.pid,birth=p.create_time());save(base/'state.json',state)
  while child.poll() is None:
   members=[]
   try:processes=[p]+p.children(recursive=True)
   except psutil.NoSuchProcess:processes=[]
   for q in processes:
    try:
     birth=q.create_time();assert state['members'].get(str(q.pid),birth)==birth;state['members'][str(q.pid)]=birth
     if q.status()==psutil.STATUS_ZOMBIE:continue
     user_owned=q.uids().effective==own.uids().effective
     native=user_owned and q.name().startswith('python') and str(app/'runtime/native.py') in q.cmdline()
     affinity=sorted(os.sched_getaffinity(q.pid));threads=[]
     for thread in q.threads():
      try:threads.append(sorted(os.sched_getaffinity(thread.id)))
      except ProcessLookupError:pass
     if native:
      identity=dict(pid=q.pid,birth=birth)
      assert 'target' not in state or state['target']==identity
      state['target']=identity
      assert affinity==[2] and all(a==[2] for a in threads)
      if (base/'requests/000.json').exists() and not (base/'target.maps').exists():
       (base/'target.maps').write_text(Path('/proc') .joinpath(str(q.pid),'maps').read_text())
     else:
      assert affinity==[0] or (user_owned and affinity==[2])
      assert all(a==affinity for a in threads)
     members.append(dict(pid=q.pid,birth=birth,native=native,rss=q.memory_info().rss,affinity=affinity,threads=threads))
    except psutil.NoSuchProcess:continue
   sample=dict(seconds=time.monotonic()-started,rss=sum(m['rss'] for m in members),members=members,
    available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free,output=sum(f.stat().st_size for f in base.rglob('*') if f.is_file()))
   log.write(json.dumps(sample)+'\\n');log.flush();state['samples']+=1;save(base/'state.json',state)
   assert sample['seconds']<300 and sample['rss']<6*1024**3 and sample['available']>=1024**3 and sample['tmpfs']>=1024**3 and sample['output']<512*1024**2
   time.sleep(.5)
  state['code']=child.wait();assert state['code']==0
 assert all(not live(dict(pid=int(pid),birth=birth)) for pid,birth in state['members'].items())
 after=account.snapshot();save(base/'cpu-after.json',after)
 state['accounting']=account.foreign_fraction(before,after,own.pid)
 assert state['accounting']['valid'] and state['accounting']['foreign_cpu_fraction']<=.01
 assert (base/'target.maps').is_file() and not (base/'perf.data').is_symlink()
 assert (base/'perf.data').resolve().parent==base.resolve()
 subprocess.run(['sudo','-n','chmod','0644',str(base/'perf.data')],check=True,timeout=10)
 with (base/'perf.script').open('x') as stdout,(base/'export.stderr').open('x') as stderr:
  p=subprocess.run(['/usr/bin/perf','script','-i',str(base/'perf.data'),'--ns','-F','pid,tid,time,period,ip,dso'],
   stdout=stdout,stderr=stderr,env=env,timeout=60)
  assert p.returncode==0
except BaseException:
 state.update(code=1,error=traceback.format_exc())
 for pid,birth in reversed(list(state['members'].items())):
  if live(dict(pid=int(pid),birth=birth)):
   subprocess.run(['sudo','-n','kill','-TERM',pid],capture_output=True,timeout=10)
 if child is not None:child.wait(timeout=15)
finally:
 state.update(complete=True,seconds=time.monotonic()-started);save(base/'state.json',state)
print(json.dumps(dict(state=state,files={{p.relative_to(base).as_posix():pin(p) for p in base.rglob('*') if p.is_file()}})))
'''
    ast.parse(script)
    (BASE/'supervisor.py').write_text(script, encoding='utf8')
    with (BASE/'terminal.json').open('xb') as stdout, (BASE/'transport.stderr').open('x') as stderr:
        process = subprocess.run(SSH+['python3','-B','-'],input=script.encode(),stdout=stdout,stderr=stderr,
                                 timeout=420,creationflags=subprocess.CREATE_NO_WINDOW)
    assert process.returncode == 0, 'Inspect recorded owner; do not relaunch after a transport failure'
    terminal = read(BASE/'terminal.json')
    print(json.dumps({k:terminal['state'].get(k) for k in ['complete','code','target','error','samples','seconds']}))


def collect():
    terminal = read(BASE/'terminal.json'); assert terminal['state']['complete']
    assert not (BASE/'collected').exists()
    source = f'''import sys,os,json,tarfile
from pathlib import Path
sys.path.insert(0,{SITE!r})
sys.path.insert(0,'/dev/shm/lokad-parakeet-ort-diagnosis-20260924')
from remote import live,pin
os.sched_setaffinity(0,{{0}})
base=Path({REMOTE!r})
state={terminal['state']!r}
assert not live(state['supervisor']) and all(not live(dict(pid=int(p),birth=b)) for p,b in state['members'].items())
files={terminal['files']!r}
for name,wanted in files.items():assert pin(base/name)==wanted
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',dereference=True) as archive:
 for name in files:archive.add(base/name,arcname=name,recursive=False)
'''
    with (BASE/'results.tar.gz').open('xb') as stdout,(BASE/'collection.stderr').open('x') as stderr:
        p = subprocess.run(SSH+['python3','-B','-'],input=source.encode(),stdout=stdout,stderr=stderr,
                           timeout=300,creationflags=subprocess.CREATE_NO_WINDOW)
    assert p.returncode == 0
    target = BASE/'collected'; target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as archive:
        members = archive.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        archive.extractall(target,filter='data')
    for name,wanted in terminal['files'].items():
        assert pin(target/name) == wanted
    write(BASE/'transfer.json',dict(passed=True,terminal=pin(BASE/'terminal.json'),archive=pin(BASE/'results.tar.gz')))
    print(json.dumps(dict(passed=True,code=terminal['state']['code'],files=len(terminal['files']))))


if __name__ == '__main__':
    assert sys.argv[1:] in ([],['collect'])
    collect() if len(sys.argv)>1 else main()
