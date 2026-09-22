"""One bounded AMD owner for correctness and four conditioned kernel processes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil

BASE = Path(__file__).resolve().parents[1]
DOTNET = '/home/vermorel/.dotnet/dotnet'
LIMITS = dict(seconds=900,preflight_available=8*1024**3,preflight_tmpfs=3*1024**3,rss=2*1024**3,
    available=1024**3,tmpfs=1024**3,artifacts=1024**3)


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))


def pin(p):
    p = Path(p)
    with p.open('rb') as f: return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def save(p,v):
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n'); t.replace(p)


def live(i):
    try:
        p = psutil.Process(i['pid']); return p.create_time()==i['birth'] and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess: return False


def clean_env():
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    env.pop('PYTHONOPTIMIZE',None)
    env.update(TMPDIR=str(BASE / 'tmp'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    return env


def verify():
    spec = read(BASE / 'payload.json'); assert spec['limits']==LIMITS
    for name,wanted in spec['files'].items():
        p = (BASE / name).resolve(); assert p.is_relative_to(BASE) and p!=BASE
        assert pin(p)==wanted,name
    for name,wanted in spec['external'].items(): assert pin(name)==wanted,name
    return spec


def idle():
    own = psutil.Process(); ancestors = {p.pid for p in own.parents()} | {own.pid}
    active = []
    for p in psutil.process_iter(['pid','name','create_time','cmdline']):
        if p.pid in ancestors: continue
        name,command = p.info['name'],' '.join(p.info['cmdline'] or [])
        if name in ('dotnet','perf') or (name.startswith('python') and ('/dev/shm/lokad-' in command or '/Onnx/' in command)):
            active.append(p.info)
    assert not active,active


def artifact_size(): return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())


def worker(state,name):
    command = [DOTNET,str(BASE / 'runtime/DirectOutputProbe.dll'),str(BASE / 'shapes.json'),
        name if name.startswith('validate') else name.split('-')[0],str(BASE / 'output' / (name+'.json'))]
    row = dict(name=name,command=command,complete=False,code=None,processes={},samples=0,peak_rss=0)
    state['runs'].append(row); save(BASE / 'identity.json',state)
    child = None; start = time.monotonic()
    try:
        row['preflight'] = dict(available=psutil.virtual_memory().available,tmpfs=shutil.disk_usage(BASE).free)
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        with (BASE / 'logs' / (name+'.log')).open('x') as log, (BASE / 'logs' / (name+'.samples.jsonl')).open('x') as samples:
            os.sched_setaffinity(0,{2})
            try:
                env = clean_env()
                if name=='validate-no-avx2': env['DOTNET_EnableAVX2']='0'
                child = subprocess.Popen(command,cwd=BASE,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            finally: os.sched_setaffinity(0,{0})
            process = psutil.Process(child.pid); identity = dict(pid=process.pid,birth=process.create_time())
            row['processes']['target'] = identity; save(BASE / 'identity.json',state)
            start = time.monotonic(); last_scan = -5; size = artifact_size()
            while child.poll() is None:
                members = []
                try:
                    assert process.create_time()==identity['birth'] and not process.children(recursive=True)
                    threads = []
                    for t in process.threads():
                        try: threads.append(dict(id=t.id,affinity=sorted(os.sched_getaffinity(t.id))))
                        except ProcessLookupError: pass
                    members.append(dict(**identity,rss=process.memory_info().rss,affinity=process.cpu_affinity(),threads=threads))
                except psutil.NoSuchProcess:
                    if child.poll() is not None: break
                    raise
                elapsed = time.monotonic()-start
                if elapsed-last_scan>=5: size=artifact_size(); last_scan=elapsed
                sample = dict(seconds=elapsed,rss=sum(p['rss'] for p in members),members=members,
                    available=psutil.virtual_memory().available,tmpfs=shutil.disk_usage(BASE).free,artifacts=size,
                    monitor_affinity=sorted(os.sched_getaffinity(0)))
                samples.write(json.dumps(sample)+'\n'); samples.flush()
                row['samples']+=1; row['peak_rss']=max(row['peak_rss'],sample['rss']); save(BASE / 'identity.json',state)
                assert elapsed<LIMITS['seconds'] and sample['rss']<LIMITS['rss']
                assert sample['available']>=LIMITS['available'] and sample['tmpfs']>=LIMITS['tmpfs'] and size<=LIMITS['artifacts']
                assert sample['monitor_affinity']==[0]
                assert all(p['affinity']==[2] and p['threads'] and all(t['affinity']==[2] for t in p['threads']) for p in members)
                time.sleep(.25)
            row['code']=child.wait(); assert row['code']==0,(name,row['code'])
        result = read(BASE / 'output' / (name+'.json'))
        assert result['passed'] and result['pid']==identity['pid'] and result['runtime']=='10.0.8'
        assert result['processor_count']==1 and result['fma']
        if name=='validate-no-avx2': assert result['flags']==['DOTNET_EnableAVX2'] and not result['avx2'] and not result['avx512']
        else: assert result['flags']==[] and result['avx2'] and result['avx512']
        print(name,'passed',flush=True)
    except BaseException:
        row['error']=traceback.format_exc()
        for i in row['processes'].values():
            if live(i): psutil.Process(i['pid']).kill()
        if child is not None: child.wait(timeout=15)
        raise
    finally:
        row.update(complete=True,seconds=time.monotonic()-start)
        if child is not None: row['code']=child.poll()
        save(BASE / 'identity.json',state)
        assert all(not live(i) for i in row['processes'].values())


def main():
    assert sys.platform=='linux' and not (BASE / 'identity.json').exists()
    os.sched_setaffinity(0,{0}); idle(); spec=verify()
    assert psutil.boot_time()==spec['boot_time']
    for folder in ['logs','output','tmp']: (BASE / folder).mkdir()
    own=psutil.Process()
    state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),
        boot_time=psutil.boot_time(),started=time.time(),runs=[])
    save(BASE / 'identity.json',state)
    try:
        for name in spec['jobs']:
            verify(); assert artifact_size()<=LIMITS['artifacts'] and time.time()-state['started']<5400
            worker(state,name)
        verify(); assert artifact_size()<=LIMITS['artifacts']; state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True,ended=time.time()); save(BASE / 'identity.json',state)
    return state['code']


if __name__=='__main__': raise SystemExit(main())
