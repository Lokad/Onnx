"""Qualify actual M43 root source, both complete SIMD suites and NuGet consumption."""
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from protocol import LIMITS, check_sample, pin, read, save, verify
from checks import inventory, suite, package, consumer, suite256

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
FLAGS=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false']
PROJECTS={'cli':BASE/'source/src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj',
          'backend':BASE/'source/tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj',
          'tensors':BASE/'source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj',
          'consumer':BASE/'consumer/PackageProbe.csproj'}


def command_for(name,spec):
    if name=='sdk-version':return [DOTNET,'--version'],True
    if name=='inventory':return [DOTNET,BASE/'bridge/Bridge.dll',BASE/'measured',BASE/'runtime',BASE/name/'instructions.json'],False
    if name=='package':return [DOTNET,'pack',BASE/'source/src/Lokad.Onnx/Lokad.Onnx.csproj','-c','Release',*FLAGS,'--no-build','--no-restore','--output',BASE/'nuget'],True
    if name=='consumer-run':
        built=read(BASE/'built.json');return [DOTNET,BASE/'consumer/bin/Release/net10.0/PackageProbe.dll',BASE/'source/tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx',built['product']['Lokad.Onnx.dll']['sha256'],BASE/name/'result.json'],False
    project,action=name.removesuffix('-256').split('-');path=PROJECTS[project]
    if action=='restore':return [DOTNET,'restore',path,*FLAGS,'--source',spec['feed'],*(['--source',BASE/'nuget'] if project=='consumer' else []),'--packages',BASE/'packages'],True
    if action=='build':return [DOTNET,'build',path,'-c','Release',*FLAGS,'--no-restore','--disable-build-servers'],True
    assert action=='tests'
    return [DOTNET,'test',path,'-c','Release',*FLAGS,'--no-build','--no-restore','--logger','trx;LogFileName='+project+'.trx','--results-directory',BASE/name],True


def after(name,spec,row):
    if name=='sdk-version':assert (BASE/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    if name=='tensors-build':
        source=PROJECTS['cli'].parent/'bin/Release/net10.0';shutil.copytree(source,BASE/'runtime')
        product={name:pin(BASE/'runtime'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
        for project in ['backend','tensors']:
            folder=PROJECTS[project].parent/'bin/Release/net10.0'
            assert pin(folder/'Lokad.Onnx.dll')==product['Lokad.Onnx.dll']
            if project=='backend':assert pin(folder/'Lokad.Onnx.Data.dll')==product['Lokad.Onnx.Data.dll']
            for p in folder.glob('Lokad.Onnx.*Tests.*'):shutil.copy2(p,BASE/'built'/p.name)
        files={p.relative_to(BASE).as_posix():pin(p) for folder in [BASE/'runtime',BASE/'built',*[PROJECTS[key].parent/'bin' for key in ['cli','backend','tensors']]] for p in folder.rglob('*') if p.is_file()}
        save(BASE/'built.json',dict(passed=True,product=product,files=files))
    if name=='inventory':
        value=inventory(read(BASE/name/'instructions.json'),spec['measured'],read(BASE/'built.json')['product']);save(BASE/name/'review.json',value)
    if name in ['backend-tests','tensors-tests']:
        key=name.split('-')[0];save(BASE/name/'review.json',suite(BASE/name/(key+'.trx'),key,BASE/'evidence'))
    if name.endswith('-tests-256'):
        key=name.split('-')[0];save(BASE/name/'review.json',suite256(BASE/name/(key+'.trx'),key,BASE/'evidence'))
    if name=='package':save(BASE/name/'review.json',package(BASE/'nuget/Lokad.Onnx.0.2.0.nupkg',read(BASE/'built.json')['product']['Lokad.Onnx.dll']))
    if name=='consumer-build':
        folder=BASE/'consumer/bin/Release/net10.0';assert pin(folder/'Lokad.Onnx.dll')==read(BASE/'built.json')['product']['Lokad.Onnx.dll']
        for suffix in ['dll','deps.json','runtimeconfig.json']:shutil.copy2(folder/('PackageProbe.'+suffix),BASE/'built'/('PackageProbe.'+suffix))
        save(BASE/'consumer-built.json',dict(passed=True,files={p.relative_to(BASE).as_posix():pin(p) for p in folder.rglob('*') if p.is_file()},executable=pin(folder/'PackageProbe.dll')))
    if name=='consumer-run':
        value=read(BASE/name/'result.json');assert value['pid']==row['child']['pid']
        result=consumer(value,read(BASE/'built.json')['product']['Lokad.Onnx.dll'],read(BASE/'consumer-built.json')['executable'],pin(BASE/'source/tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx'))
        save(BASE/name/'review.json',result)


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
                        if name.endswith('-tests-256'):job_env['DOTNET_EnableAVX512']='0'
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
