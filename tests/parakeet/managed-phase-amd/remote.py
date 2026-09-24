"""Bounded VM build and capture; each launch has its own terminal state."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil

BASE=Path(__file__).resolve().parent
DOTNET='/home/vermorel/.dotnet/dotnet'
GIB=1024**3


def pin(path):
    path=Path(path)
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def live(identity):
    try:
        p=psutil.Process(identity['pid']);return p.create_time()==identity['birth'] and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False


def idle():
    own=psutil.Process();parents={own.pid,*[p.pid for p in own.parents()]}
    for p in psutil.process_iter(['name','cmdline']):
        if p.pid in parents:continue
        command=' '.join(p.info['cmdline'] or [])
        assert p.info['name'] not in ['dotnet','perf'],p.info
        assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command),p.info


def verify():
    spec=read(BASE/'spec.json');assert psutil.boot_time()==spec['boot']
    for name,wanted in spec['files'].items():assert pin(BASE/name)==wanted,name
    for name,wanted in spec['external'].items():assert pin(name)==wanted,name
    return spec


def total_size():return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())


def job(state,name,command,environment,cwd,limits,spec,output=None):
    own=psutil.Process();child=None
    preflight=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(BASE).free)
    row=dict(name=name,command=list(map(str,command)),complete=False,code=None,members={},samples=0,preflight=preflight)
    state['runs'].append(row);save(BASE/(state['kind']+'-state.json'),state)
    assert preflight['available']>=limits['available_before'] and preflight['tmpfs']>=limits['tmpfs_before']
    start=time.monotonic()
    try:
        with (BASE/'logs'/(name+'.stdout')).open('x') as out,(BASE/'logs'/(name+'.stderr')).open('x') as err,(BASE/'logs'/(name+'.resources.jsonl')).open('x') as log:
            own.cpu_affinity([2])
            try:child=subprocess.Popen(list(map(str,command)),cwd=cwd,env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            finally:own.cpu_affinity([0])
            process=psutil.Process(child.pid);row['owner']=dict(pid=child.pid,birth=process.create_time())
            row['members'][str(child.pid)]=row['owner']['birth']
            save(BASE/(state['kind']+'-state.json'),state)
            while child.poll() is None:
                if output is not None and (output/'ready.json').exists() and not (output/'release.json').exists():
                    ready=read(output/'ready.json')
                    assert ready['pid']==child.pid and abs(ready['birth_milliseconds']/1000-row['owner']['birth'])<1.1
                    assert ready['warmup_records']==20 and ready['affinity']==4 and not ready['flags']
                    save(output/'release.json',dict(pid=child.pid,sampled=False));row['ready']=ready
                members=[]
                try:
                    for p in [process]+process.children(recursive=True):
                        try:
                            birth=p.create_time();assert row['members'].get(str(p.pid),birth)==birth
                            row['members'][str(p.pid)]=birth
                            if p.status()==psutil.STATUS_ZOMBIE:continue
                            threads=[]
                            for t in p.threads():
                                try:threads.append(sorted(os.sched_getaffinity(t.id)))
                                except ProcessLookupError:pass
                            members.append(dict(pid=p.pid,birth=birth,rss=p.memory_info().rss,affinity=p.cpu_affinity(),threads=threads))
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                sample=dict(seconds=time.monotonic()-start,members=members,rss=sum(m['rss'] for m in members),
                    available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(BASE).free,output=total_size())
                log.write(json.dumps(sample)+'\n');log.flush();row['samples']+=1
                save(BASE/(state['kind']+'-state.json'),state)
                assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss']
                assert sample['available']>=spec['minimum_free'] and sample['tmpfs']>=spec['minimum_free'] and sample['output']<spec['output_limit']
                assert all(m['affinity']==[2] and all(t==[2] for t in m['threads']) for m in members)
                time.sleep(.5)
            row['code']=child.wait();assert row['code']==0,(name,row['code'])
        assert all(not live(dict(pid=int(p),birth=b)) for p,b in row['members'].items())
    except BaseException:
        for pid,birth in reversed(list(row['members'].items())):
            if live(dict(pid=int(pid),birth=birth)):psutil.Process(int(pid)).kill()
        if child is not None:child.wait(timeout=15)
        raise
    finally:
        row.update(complete=True,seconds=time.monotonic()-start,code=None if child is None else child.poll())
        save(BASE/(state['kind']+'-state.json'),state)


def build(state,env,spec):
    for name in ['tmp','cli-home','packages','http-cache']:(BASE/name).mkdir()
    env=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',
        NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0',TMPDIR=str(BASE/'tmp'))
    flags=['--tl:off','--nologo','-v','minimal','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:FrozenProductDirectory='+spec['prior']]
    limits=spec['build_limits']
    job(state,'sdk-version',[DOTNET,'--version'],env,BASE/'data-source',limits,spec)
    assert (BASE/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name,project in [('data','ObserverData'),('consumer','SampledAudio'),('bridge','Bridge')]:
        folder=BASE/(name+'-source');path=folder/(project+'.csproj')
        job(state,name+'-restore',[DOTNET,'restore',path,*flags,'--source','/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed','--packages',BASE/'packages'],env,folder,limits,spec)
        job(state,name+'-build',[DOTNET,'build',path,'-c','Release',*flags,'--no-restore','--disable-build-servers'],env,folder,limits,spec)
    for name in ['runtime-control','runtime-observed']:
        folder=BASE/name;folder.mkdir()
        for original in Path(spec['prior']).iterdir():
            if not original.is_file():continue
            source=original
            if original.name.startswith('SampledAudio.'):
                source=BASE/'consumer-source/bin/Release/net10.0'/original.name
            elif original.name=='Lokad.Onnx.Data.dll' and name=='runtime-observed':
                source=BASE/'data-source/bin/Release/net10.0/Lokad.Onnx.Data.dll'
            shutil.copy2(source,folder/original.name)
        assert pin(folder/'Lokad.Onnx.dll')==spec['core']
    (BASE/'inventory').mkdir()
    job(state,'inventory',[DOTNET,BASE/'bridge-source/bin/Release/net10.0/Bridge.dll',spec['prior'],BASE/'runtime-observed',BASE/'inventory/instructions.json'],env,BASE,limits,spec)
    save(BASE/'built.json',dict(core=pin(BASE/'runtime-observed/Lokad.Onnx.dll'),data=pin(BASE/'runtime-observed/Lokad.Onnx.Data.dll'),
        consumer=pin(BASE/'runtime-observed/SampledAudio.dll'),runtime_files={p.relative_to(BASE).as_posix():pin(p) for folder in ['runtime-control','runtime-observed'] for p in (BASE/folder).iterdir()}))


def capture(state,env,spec):
    approval=read(BASE/'build-review.json');assert approval['passed'] and approval['built']==pin(BASE/'built.json')
    built=read(BASE/'built.json')
    for name,wanted in built['runtime_files'].items():assert pin(BASE/name)==wanted
    app=Path(spec['app'])
    module=importlib.util.spec_from_file_location('cpu_accounting',app/'runtime/campaign_processes.py')
    accounting=importlib.util.module_from_spec(module);module.loader.exec_module(accounting)
    for mode in ['control','phase','wall']:
        verify()
        runtime=BASE/('runtime-control' if mode=='control' else 'runtime-observed')
        output=BASE/mode
        environment=dict(env,PARAKEET_PHASE_MODE=mode,PARAKEET_PHASE_DATA_SHA=pin(runtime/'Lokad.Onnx.Data.dll')['sha256'])
        before=accounting.snapshot()
        job(state,mode,[DOTNET,runtime/'SampledAudio.dll',app/'assets',app/'manifests/current-parakeet.json',output,'timing','control'],environment,BASE,spec['capture_limits'],spec,output)
        after=accounting.snapshot();row=state['runs'][-1]
        row['cpu_before']=before;row['cpu_after']=after;row['accounting']=accounting.foreign_fraction(before,after,state['supervisor']['pid'])
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction']<=.01
        save(BASE/'capture-state.json',state)
        value=read(output/'result.json');assert value['passed'] and len(value['records'])==80


def main():
    own=psutil.Process();own.cpu_affinity([0]);idle();kind=sys.argv[1];assert kind in ['build','capture']
    spec=verify();path=BASE/(kind+'-state.json');assert not path.exists()
    (BASE/'logs').mkdir(exist_ok=True)
    state=dict(kind=kind,complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[],started=time.time())
    save(path,state)
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_','parakeet_phase_'))}
    env.pop('PYTHONOPTIMIZE',None)
    env['PATH']=str(Path(DOTNET).parent)+os.pathsep+env.get('PATH','')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='1'
    try:
        (build if kind=='build' else capture)(state,env,spec)
        verify();state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());traceback.print_exc()
    finally:
        state.update(complete=True,ended=time.time());save(path,state)
    return state['code']


if __name__=='__main__':raise SystemExit(main())
