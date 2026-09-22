"""Eight fresh workers qualify original/wide/spatial raw and captured-layer graphs."""
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from protocol import FILTER, LIMITS, check_sample, pin, read, save, verify
from checks import check_result, check_suite, check_native

BASE = Path(__file__).resolve().parents[1]


def live(identity):
    try:
        p = psutil.Process(identity['pid'])
        return p.create_time() == identity['birth'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess: return False


def size(folder): return sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())


def idle():
    own = psutil.Process(); ancestors = {own.pid,*[p.pid for p in own.parents()]}
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid in ancestors: continue
        command = ' '.join(p.info['cmdline'] or [])
        assert p.info['name'] not in ['dotnet','perf'], ('Existing runtime owner',p.pid)
        assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command), ('Existing benchmark owner',p.pid)


def main():
    assert sys.platform == 'linux' and not sys.flags.optimize and not (BASE/'identity.json').exists()
    own = psutil.Process(); own.cpu_affinity([0]); idle(); spec = verify(BASE)
    assert psutil.boot_time() == spec['boot_time']
    assert pin(Path(sys.executable)) == spec['interpreter']
    assert not live(spec['previous_owner'])
    (BASE/'logs').mkdir(); (BASE/'tmp').mkdir(); (BASE/'empty-feed').mkdir()
    state = dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),
                 started=time.time(),boot_time=psutil.boot_time(),runs=[])
    path = BASE/'identity.json'; save(path,state)
    env = {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    env.pop('PYTHONOPTIMIZE',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMPDIR=str(BASE/'tmp'))
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[key] = '1'
    try:
        campaign_started = time.monotonic()
        for name in spec['jobs']:
            assert time.monotonic()-campaign_started < 4*3600
            verify(BASE)
            if (BASE/'built.json').exists():
                for file,wanted in read(BASE/'built.json')['files'].items():assert pin(BASE/file)==wanted,file
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
                        mode,width=name.split('-')
                        job_env=dict(env)
                        if width=='256':job_env['DOTNET_EnableAVX512']='0'
                        if width=='scalar':job_env['DOTNET_EnableHWIntrinsic']='0'
                        if mode=='consumer':
                            project=BASE/'consumer/ModelReplay.csproj'
                            flags=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false']
                            if width=='restore':command=['/home/vermorel/.dotnet/dotnet','restore',str(project),*flags,'--source',str(BASE/'empty-feed'),'--packages',str(BASE/'packages')]
                            else:
                                assert width=='build'
                                command=['/home/vermorel/.dotnet/dotnet','build',str(project),'-c','Release',*flags,'--no-restore','--disable-build-servers']
                        elif mode=='suite':
                            executable=BASE/'backend/Lokad.Onnx.Backend.Tests.dll'
                            command=['/home/vermorel/.dotnet/dotnet','vstest',str(executable),'/TestCaseFilter:FullyQualifiedName~Lstm',
                                '/Logger:trx;LogFileName=suite.trx','/ResultsDirectory:'+str(BASE/name)]
                        elif mode=='native':command=[sys.executable,'-B',str(BASE/'tools/native.py'),str(BASE/name)]
                        else:
                            assert mode in ['selected','candidate']
                            job_env['DOTNET_JitDisasm']=FILTER
                            executable=BASE/'runtime'/mode/'LstmModelReplay.dll'
                            command=['/home/vermorel/.dotnet/dotnet',str(executable),str(BASE/'fixtures'),str(BASE/name/'result.json'),spec['cores'][mode]['sha256'],mode,width]
                        child = subprocess.Popen(command,cwd=BASE,
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
                            output=size(BASE/'consumer') if mode=='consumer' else size(BASE/name) if (BASE/name).exists() else 0,artifacts=size(BASE))
                        log.write(__import__('json').dumps(sample)+'\n'); log.flush()
                        row['samples'] += 1; row['peak_rss'] = max(row['peak_rss'],sample['rss']); save(path,state)
                        check_sample(sample); time.sleep(.25)
                    row['code'] = child.wait(); assert row['code'] == 0, (name,row['code'])
                assert all(not live(dict(pid=int(pid),birth=birth)) for pid,birth in row['members'].items())
                if mode=='consumer':
                    if width=='build':
                        (BASE/'built').mkdir()
                        for suffix in ['dll','deps.json','runtimeconfig.json']:
                            source=BASE/'consumer/bin/Release/net10.0'/('LstmModelReplay.'+suffix)
                            for role in ['selected','candidate']:shutil.copy2(source,BASE/'runtime'/role/source.name)
                            shutil.copy2(source,BASE/'built'/source.name)
                        spec['consumer']=pin(BASE/'runtime/selected/LstmModelReplay.dll')
                        files={p.relative_to(BASE).as_posix():pin(p) for p in (BASE/'built').iterdir()}
                        files.update({(Path('runtime')/role/p.name).as_posix():pin(p) for role in ['selected','candidate'] for p in (BASE/'built').iterdir()})
                        save(BASE/'built.json',dict(passed=True,consumer=spec['consumer'],files=files))
                elif mode=='suite':check_suite(BASE/name/'suite.trx',BASE/'ordinary.trx')
                else:
                    result=read(BASE/name/'result.json');assert result['passed'] and result['pid']==row['child']['pid']
                    if mode=='native':check_native(result,BASE/name,BASE)
                    else:
                        assert result['runtime']=='10.0.8'
                        check_result(result,mode,width,spec,BASE)


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
