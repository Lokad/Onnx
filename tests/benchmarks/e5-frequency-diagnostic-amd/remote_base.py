"""Bounded, serial builds and diagnostic trace pairs; never produces a score."""
import json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from counter import Counter, epoch, intervals
from protocol import LIMITS,PROVIDERS,check_sample,pin,read,save,verify
BASE=Path(__file__).resolve().parents[1]
DOTNET='/home/vermorel/.dotnet/dotnet'
FLAGS=['--tl:off','--nologo','-v','minimal','-p:EnableSourceControlManagerQueries=false','-p:EnableSourceLink=false','-p:UseSharedCompilation=false','-nr:false','-p:NuGetAudit=false']
def live(identity):
 try:
  p=psutil.Process(identity['pid']);return p.create_time()==identity['birth'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
def size(folder):
 total=0
 for p in folder.rglob('*'):
  try:
   if p.is_file():total+=p.stat().st_size
  except FileNotFoundError:pass
 return total
def idle():
 own=psutil.Process();ancestors={own.pid,*[p.pid for p in own.parents()]}
 for p in psutil.process_iter(['pid','name','cmdline']):
  if p.pid in ancestors:continue
  command=' '.join(p.info['cmdline'] or [])
  assert p.info['name'] not in ['dotnet','perf'],('Existing runtime owner',p.pid)
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command),('Existing benchmark owner',p.pid)
def command_for(name,spec):
 if name=='sdk-version':return [DOTNET,'--version'],True,2
 if name=='tracer-version':return [DOTNET,BASE/'tracer/dotnet-trace.dll','--version'],False,0
 if name.startswith(('producer-','exporter-')):
  kind,action=name.split('-');project=BASE/'source'/('consumer/Producer.csproj' if kind=='producer' else 'exporter/Exporter.csproj')
  command=[DOTNET,action,project,*FLAGS]
  command+=['--source',spec['feed'],'--packages',BASE/'packages'] if action=='restore' else ['-c','Release','--no-restore','--disable-build-servers']
  return command,True,2
 role,action=name.split('-')
 if action=='capture':return [DOTNET,BASE/'runtimes'/role/'ParakeetDispatchEvents.dll',BASE,role,['current','candidate'].index(role),BASE/name/'result.json'],False,2
 if action=='export':return [DOTNET,BASE/'export-runtime/DispatchEventsExport.dll',BASE/(role+'-capture/capture.nettrace'),BASE/name/'events'],False,0
 assert action=='stacks'
 return [DOTNET,BASE/'tracer/dotnet-trace.dll','convert',BASE/(role+'-capture/capture.nettrace'),'--format','Speedscope','--output',BASE/name/'speedscope'],False,0
def after(name,spec,row):
 if name=='sdk-version':assert (BASE/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
 if name in ['producer-build','exporter-build']:
  built=read(BASE/'built.json') if (BASE/'built.json').exists() else dict(passed=True,files={})
  if name=='producer-build':
   folder=BASE/'source/consumer/bin/Release/net10.0'
   for role in ['current','candidate']:
    for suffix in ['dll','deps.json','runtimeconfig.json']:
     source=folder/('ParakeetDispatchEvents.'+suffix);target=BASE/'runtimes'/role/source.name
     assert not target.exists();shutil.copy2(source,target);built['files'][target.relative_to(BASE).as_posix()]=pin(target)
   built['consumer']=pin(folder/'ParakeetDispatchEvents.dll')
  else:
   folder=BASE/'source/exporter/bin/Release/net10.0';target=BASE/'export-runtime';shutil.copytree(folder,target)
   for p in target.rglob('*'):
    if p.is_file():built['files'][p.relative_to(BASE).as_posix()]=pin(p)
   built['exporter']=pin(target/'DispatchEventsExport.dll')
  save(BASE/'built.json',built)
 if name.endswith('-capture'):
  role=name.split('-')[0];value=read(BASE/name/'result.json');ready=read(BASE/name/'ready.json');enabled=read(BASE/name/'collector-enabled.json')
  assert value['passed'] and value['diagnosticOnly'] and value['protocol']=='parakeet-dispatch-events-v1'
  assert value['pid']==row['processes']['worker']['pid']==ready['pid']==enabled['pid']
  assert value['nativeThread']==ready['native_thread'] and ready['counter']<enabled['counter']
  assert value['runtime']=='10.0.8' and value['flags']=={} and value['role']==role
  assert value['assembly']==read(BASE/'built.json')['consumer']['sha256']
  assert value['core_sha256']==spec['products'][role]['Lokad.Onnx.dll']['sha256']
  assert len(value['rows'])==21 and value['calls']==2520 and value['warmups']==value['measured']==1260
  assert (BASE/name/'capture.nettrace').stat().st_size>0
 if name.endswith('-export'):
  value=read(BASE/name/'events/summary.json');assert value['complete'] and value['lost']==0 and value['clr_events']>0
  assert value['input_sha256']==pin(BASE/(name.split('-')[0]+'-capture/capture.nettrace'))['sha256']
 if name.endswith('-stacks'):assert list((BASE/name).glob('*.speedscope.json'))
def main():
 assert sys.platform=='linux' and not sys.flags.optimize and not (BASE/'identity.json').exists()
 own=psutil.Process();own.cpu_affinity([0]);idle();spec=verify(BASE)
 assert psutil.boot_time()==spec['boot_time'] and pin(Path(sys.executable))==spec['interpreter'] and not live(spec['previous_owner'])
 for name in ['logs','tmp','packages','cli-home','http-cache']:(BASE/name).mkdir()
 state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),started=time.time(),boot_time=psutil.boot_time(),runs=[])
 path=BASE/'identity.json';save(path,state)
 env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))};env.pop('PYTHONOPTIMIZE',None)
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMPDIR=str(BASE/'tmp'));env['PATH']=str(Path(DOTNET).parent)+os.pathsep+env.get('PATH','')
 for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
 build_env=dict(env,DOTNET_CLI_HOME=str(BASE/'cli-home'),DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1',NUGET_PACKAGES=str(BASE/'packages'),NUGET_HTTP_CACHE_PATH=str(BASE/'http-cache'),MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_USE_MSBUILD_SERVER='0')
 try:
  campaign_started=time.monotonic()
  for name in spec['jobs']:
   assert time.monotonic()-campaign_started<4*3600;verify(BASE)
   if (BASE/'built.json').exists():
    for file,wanted in read(BASE/'built.json')['files'].items():assert pin(BASE/file)==wanted,file
   waiting=time.monotonic();observations=[]
   while True:
    sample=dict(seconds=time.monotonic()-waiting,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(BASE).free)
    observations.append(sample);save(BASE/(name+'-preflight.json'),observations)
    assert sample['seconds']<900 and sample['tmpfs']>=LIMITS['preflight_tmpfs']
    if sample['available']>=LIMITS['preflight_available']:break
    time.sleep(10)
   (BASE/name).mkdir();row=dict(name=name,complete=False,code=None,preflight=observations[-1],preflight_observations=observations,members={},affinities={},processes={},samples=0,peak_rss=0,commands={},exitcodes={})
   state['runs'].append(row);save(path,state);children={};handles=[];started=time.monotonic();counter=None;frequency_done=False
   def spawn(label,command,job_env,cpu):
    out=(BASE/'logs'/(name+('' if label=='worker' else '-'+label)+'.stdout')).open('x');err=(BASE/'logs'/(name+('' if label=='worker' else '-'+label)+'.stderr')).open('x');handles.extend([out,err])
    row['commands'][label]=list(map(str,command));save(path,state);own.cpu_affinity([cpu])
    try:child=subprocess.Popen(list(map(str,command)),cwd=BASE/'source',env=job_env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    finally:own.cpu_affinity([0])
    children[label]=child;process=psutil.Process(child.pid);identity=dict(pid=child.pid,birth=process.create_time(),affinity=[cpu]);row['processes'][label]=identity
    row['members'][str(child.pid)]=identity['birth'];row['affinities'][str(child.pid)]=[cpu];save(path,state)
   try:
    if name.endswith('-capture'):
     counter=Counter(BASE/name/'counters');row['anchors']=[]
     before=time.monotonic_ns();spawn('frequency',counter.command(),env,0)
     row['anchors'].append(counter.begin(before));save(path,state)
    command,build,cpu=command_for(name,spec);spawn('worker',command,build_env if build else env,cpu)
    with (BASE/'logs'/(name+'.jsonl')).open('x') as log:
     while any(child.poll() is None for child in children.values()):
      if counter is not None and not frequency_done:
       if children['worker'].poll() is not None:
        row['anchors'].append(counter.finish(children['frequency']))
        row['epoch']=epoch(row['anchors']);row['counter_intervals']=len(intervals(counter.output.read_text()))
        frequency_done=True;save(path,state)
       else:assert children['frequency'].poll() is None,'Frequency collector ended before workload'
      if name.endswith('-capture') and 'collector' not in children and (BASE/name/'ready.json').exists():
       ready=read(BASE/name/'ready.json');identity=row['processes']['worker'];assert ready['pid']==identity['pid'] and live(identity)
       spawn('collector',[DOTNET,BASE/'tracer/dotnet-trace.dll','collect','--process-id',identity['pid'],'--providers',PROVIDERS,'--buffersize','64','--duration','00:00:15:00','--output',BASE/name/'capture.nettrace'],env,0)
      members=[]
      for label,child in list(children.items()):
       if child.poll() is not None:continue
       identity=row['processes'][label]
       try:
        process=psutil.Process(child.pid);assert process.create_time()==identity['birth']
        for p in [process]+process.children(recursive=True):
         try:
          birth=p.create_time();assert row['members'].get(str(p.pid),birth)==birth;row['members'][str(p.pid)]=birth;row['affinities'][str(p.pid)]=identity['affinity']
          if p.status()==psutil.STATUS_ZOMBIE:continue
          threads=[]
          for t in p.threads():
           try:threads.append(dict(tid=t.id,affinity=sorted(os.sched_getaffinity(t.id))))
           except ProcessLookupError:pass
          if not threads and not live(dict(pid=p.pid,birth=birth)):continue
          members.append(dict(role=label,pid=p.pid,birth=birth,rss=p.memory_info().rss,affinity=p.cpu_affinity(),expected_affinity=identity['affinity'],threads=threads))
         except psutil.NoSuchProcess:pass
       except psutil.NoSuchProcess:pass
      sample=dict(seconds=time.monotonic()-started,members=members,rss=sum(m['rss'] for m in members),available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(BASE).free,output=size(BASE/name),artifacts=size(BASE))
      log.write(json.dumps(sample)+'\n');log.flush();row['samples']+=1;row['peak_rss']=max(row['peak_rss'],sample['rss']);save(path,state);check_sample(sample);time.sleep(.25)
    row['exitcodes']={label:child.wait() for label,child in children.items()};assert all(code==0 for code in row['exitcodes'].values()),row['exitcodes']
    if name.endswith('-capture'):assert set(children)=={'worker','collector','frequency'} and frequency_done
    assert all(not live(dict(pid=int(pid),birth=birth)) for pid,birth in row['members'].items());after(name,spec,row);row['code']=0
   except BaseException:
    row['error']=traceback.format_exc()
    for label,child in children.items():
     if child.poll() is None:
      try:
       parent=psutil.Process(child.pid)
       assert parent.create_time()==row['processes'][label]['birth']
       for p in parent.children(recursive=True):
        row['members'][str(p.pid)]=p.create_time();row['affinities'][str(p.pid)]=row['processes'][label]['affinity']
      except psutil.NoSuchProcess:pass
    if counter is not None and not (counter.folder/'stop').exists():counter.stop()
    if 'frequency' in children:
     try:children['frequency'].wait(timeout=5)
     except subprocess.TimeoutExpired:pass
    for pid,birth in reversed(list(row['members'].items())):
     if live(dict(pid=int(pid),birth=birth)):
      try:psutil.Process(int(pid)).kill()
      except psutil.NoSuchProcess:pass
      except psutil.AccessDenied:
       if live(dict(pid=int(pid),birth=birth)):subprocess.run(['sudo','-n','/bin/kill','-KILL',pid],check=True,timeout=5)
    for child in children.values():child.wait(timeout=15)
    row['exitcodes']={label:child.poll() for label,child in children.items()};row['code']=1;raise
   finally:
    if counter is not None:counter.close()
    for handle in handles:handle.close()
    row.update(complete=True,seconds=time.monotonic()-started);save(path,state)
   print(name,'passed',flush=True)
  verify(BASE);assert size(BASE)<=LIMITS['artifacts'];state['code']=0
 except BaseException:state.update(code=1,error=traceback.format_exc());traceback.print_exc()
 finally:state.update(complete=True,ended=time.time());save(path,state)
 return state['code']
if __name__=='__main__':raise SystemExit(main())
