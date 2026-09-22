"""Run exact qualified consumers against current and M26 products under fixed bounds."""
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from protocol import LIMITS, check_sample, pin, read, save, verify
from checks import check_result

BASE = Path(__file__).resolve().parents[1]


def live(identity):
    try:
        p = psutil.Process(identity['pid'])
        return p.create_time() == identity['birth'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess: return False


def size(folder):
    total=0
    for p in folder.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:pass
    return total


def idle():
    own = psutil.Process(); ancestors = {own.pid,*[p.pid for p in own.parents()]}
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid in ancestors: continue
        command = ' '.join(p.info['cmdline'] or [])
        assert p.info['name'] not in ['dotnet','perf'], ('Existing runtime owner',p.pid)
        assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command), ('Existing benchmark owner',p.pid)


DOTNET='/home/vermorel/.dotnet/dotnet'

def command_for(name,spec):
    role,width=name.split('-');assert role in ['selected','candidate'] and width in ['qualify','0','1','2','3']
    return [DOTNET,BASE/'runtime'/role/'LstmScreen.dll',BASE/'fixtures',BASE/name,spec['cores'][role]['sha256'],role,'qualify' if width=='qualify' else 'time'],False

def after(name,spec,row):
    role,width=name.split('-');result=read(BASE/name/'result.json')
    assert result['pid']==row['child']['pid'] and result['runtime']=='10.0.8'
    save(BASE/name/'review.json',check_result(result,name,spec,BASE/name,read(BASE/'references/capture.json')))


def main():
    assert sys.platform == 'linux' and not sys.flags.optimize and not (BASE/'identity.json').exists()
    own = psutil.Process(); own.cpu_affinity([0]); idle(); spec = verify(BASE)
    assert psutil.boot_time() == spec['boot_time']
    assert pin(Path(sys.executable)) == spec['interpreter']
    assert not live(spec['previous_owner'])
    for name in ['logs','tmp','packages','cli-home','http-cache','nuget','built']:(BASE/name).mkdir()
    state = dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),
                 started=time.time(),boot_time=psutil.boot_time(),runs=[])
    path = BASE/'identity.json'; save(path,state)
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    env.pop('PYTHONOPTIMIZE',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMPDIR=str(BASE/'tmp'))
    env['PATH']=str(Path(DOTNET).parent)+os.pathsep+env.get('PATH','')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[key] = '1'
    build_env=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0')
    try:
        campaign_started = time.monotonic()
        for name in spec['jobs']:
            assert time.monotonic()-campaign_started < 4*3600
            verify(BASE)
            if (BASE/'built.json').exists():
                for file,wanted in read(BASE/'built.json')['files'].items():assert pin(BASE/file)==wanted,file
            if (BASE/'consumer-built.json').exists():
                for file,wanted in read(BASE/'consumer-built.json')['files'].items():assert pin(BASE/file)==wanted,file
            waiting = time.monotonic(); observations = []
            while True:
                sample = dict(seconds=time.monotonic()-waiting,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(BASE).free)
                observations.append(sample); save(BASE/(name+'-preflight.json'),observations)
                assert sample['seconds'] < 900 and sample['tmpfs'] >= LIMITS['preflight_tmpfs']
                if sample['available'] >= LIMITS['preflight_available']: break
                time.sleep(10)
            (BASE/name).mkdir()
            row = dict(name=name,complete=False,code=None,preflight=observations[-1],preflight_observations=observations,
                       members={},samples=0,peak_rss=0)
            state['runs'].append(row); save(path,state); child = None; started = time.monotonic()
            try:
                with (BASE/'logs'/(name+'.stdout')).open('x') as out, (BASE/'logs'/(name+'.stderr')).open('x') as err, (BASE/'logs'/(name+'.jsonl')).open('x') as log:
                    own.cpu_affinity([2])
                    try:
                        command,build=command_for(name,spec)
                        job_env=dict(build_env if build else env)
                        row['command']=list(map(str,command));save(path,state)
                        child = subprocess.Popen(list(map(str,command)),cwd=BASE/'source',
                            env=job_env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                    finally: own.cpu_affinity([0])
                    process = psutil.Process(child.pid)
                    row['child'] = dict(pid=child.pid,birth=process.create_time()); row['members'][str(child.pid)] = row['child']['birth']; save(path,state)
                    while child.poll() is None:
                        members = []
                        try:
                            for p in [process]+process.children(recursive=True):
                                try:
                                    birth = p.create_time(); assert row['members'].get(str(p.pid),birth) == birth
                                    row['members'][str(p.pid)] = birth
                                    if p.status() == psutil.STATUS_ZOMBIE: continue
                                    threads = []
                                    for t in p.threads():
                                        try: threads.append(dict(tid=t.id,affinity=sorted(os.sched_getaffinity(t.id))))
                                        except ProcessLookupError: pass
                                    if not threads and not live(dict(pid=p.pid,birth=birth)): continue
                                    members.append(dict(pid=p.pid,birth=birth,rss=p.memory_info().rss,affinity=p.cpu_affinity(),threads=threads))
                                except psutil.NoSuchProcess: pass
                        except psutil.NoSuchProcess: pass
                        sample = dict(seconds=time.monotonic()-started,members=members,rss=sum(m['rss'] for m in members),
                            available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(BASE).free,
                            output=size(BASE/name),artifacts=size(BASE))
                        log.write(__import__('json').dumps(sample)+'\n'); log.flush()
                        row['samples'] += 1; row['peak_rss'] = max(row['peak_rss'],sample['rss']); save(path,state)
                        check_sample(sample); time.sleep(.25)
                    row['code'] = child.wait(); assert row['code'] == 0, (name,row['code'])
                assert all(not live(dict(pid=int(pid),birth=birth)) for pid,birth in row['members'].items())
                after(name,spec,row)



            except BaseException:
                row['error'] = traceback.format_exc()
                for pid,birth in reversed(list(row['members'].items())):
                    if live(dict(pid=int(pid),birth=birth)):
                        try: psutil.Process(int(pid)).kill()
                        except psutil.NoSuchProcess: pass
                if child is not None: child.wait(timeout=15)
                raise
            finally:
                row.update(complete=True,code=None if child is None else child.poll(),seconds=time.monotonic()-started); save(path,state)
            print(name,'passed',flush=True)
        verify(BASE); assert size(BASE) <= LIMITS['artifacts']; state['code'] = 0
    except BaseException:
        state.update(code=1,error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True,ended=time.time()); save(path,state)
    return state['code']


if __name__ == '__main__': raise SystemExit(main())
