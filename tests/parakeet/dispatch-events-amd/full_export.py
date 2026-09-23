"""Export ALL records from the closed M42 traces, with no inference rerun."""
import base64,collections,json,shutil,sys,tarfile
from pathlib import Path
import run as capture_run
from protocol import LIMITS,pin,read,save,check_sample
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
CAPTURE=ROOT/'artifacts/parakeet-dispatch-events-amd-20260923'
BASE=ROOT/'artifacts/parakeet-dispatch-full-export-amd-20260923'
REMOTE='/dev/shm/lokad-parakeet-dispatch-full-export-20260923'
OLDREMOTE='/dev/shm/lokad-parakeet-dispatch-events-20260923'
JOBS=['sdk-version','exporter-restore','exporter-build','current-export','candidate-export']
capture_run.BASE=BASE;capture_run.REMOTE=REMOTE
capture_run.PRELUDE=capture_run.PRELUDE.replace(OLDREMOTE,REMOTE)
def previous_closed():
 assert pin(CAPTURE/'closed.json')['sha256']=='c6e1e4d42d3f377f77382a4091ad1150c1a62335a72c7d63a38c728d7a753e19'
 for name,wanted in read(CAPTURE/'closed.json')['files'].items():assert pin(CAPTURE/name)==wanted,name
 from prepare import previous_closed as root_check
 root_check()
capture_run.previous_closed=previous_closed
def prepare():
 previous_closed();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
 def copy(source,target):
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);originals[source.relative_to(ROOT).as_posix()]=pin(source)
 copy(TOOLS/'ExportAll.cs.txt',bundle/'source/exporter/Export.cs')
 copy(TOOLS/'Exporter.csproj',bundle/'source/exporter/Exporter.csproj')
 copy(ROOT/'global.json',bundle/'source/global.json')
 for name in ['protocol.py','remote.py']:copy(TOOLS/name,bundle/'tools'/('remote_base.py' if name=='remote.py' else name))
 (bundle/'tools/remote.py').write_text('import protocol\nprotocol.JOBS='+repr(JOBS)+'\nfrom remote_base import *\nif __name__=="__main__":raise SystemExit(main())\n')
 copy(CAPTURE/'closed.json',bundle/'evidence/capture-closed.json')
 copy(CAPTURE/'collected/collection.json',bundle/'evidence/capture-collection.json')
 for name in ['full_export.py','ExportAll.cs.txt']:originals[(TOOLS/name).relative_to(ROOT).as_posix()]=pin(TOOLS/name)
 shutil.copy2(ROOT/'.agent/m42-parakeet-dispatch-events-20260923.md',bundle/'prospective-plan.md')
 links={}
 payload=read(CAPTURE/'payload.json')
 for name,wanted in payload['files'].items():
  if name.startswith('tracer/'):links[name]=dict(source=OLDREMOTE+'/'+name,identity=wanted)
 for role in ['current','candidate']:
  name=role+'-capture/capture.nettrace';links[name]=dict(source=OLDREMOTE+'/'+name,identity=pin(CAPTURE/'collected'/name))
 save(bundle/'stage.json',dict(passed=True,links=links,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
 with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
  for p in sorted(bundle.rglob('*')):
   if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
 save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
 print(json.dumps(dict(passed=True,archive=pin(BASE/'payload.tar.gz'),links=len(links))))
def stage():
 spec=capture_run.prepared();assert not (BASE/'staged.json').exists()
 # This small transport embeds sources only; existing trace/tracer bytes are linked.
 encoded=base64.b64encode((BASE/'payload.tar.gz').read_bytes()).decode()
 extracted=json.loads(capture_run.ssh(capture_run.PRELUDE+f'''
import base64,io
assert not base.exists();base.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r}))) as tar:
 members=tar.getmembers();assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 assert len({{m.name for m in members}})==len(members);tar.extractall(base,filter='data')
from protocol import read,pin
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0 and pin(base/'stage.json')=={spec['stage']!r}
stage=read(base/'stage.json');receipt=read(base/'evidence/capture-collection.json')
assert receipt['terminal'] and receipt['code']==0 and not any(live(i) for i in receipt['identities'])
assert pin(Path({OLDREMOTE!r})/'collection.json')==pin(base/'evidence/capture-collection.json')
for name,wanted in stage['files'].items():assert pin(base/name)==wanted,name
for name,link in stage['links'].items():
 target=(base/name).resolve();assert target.is_relative_to(base.resolve()) and not target.exists()
 source=Path(link['source']);assert pin(source)==link['identity'];target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
print(json.dumps(dict(passed=True,files=len(stage['files']),links=len(stage['links']))))
'''))
 save(BASE/'extracted.json',extracted)
 result=json.loads(capture_run.ssh(capture_run.PRELUDE+f'''
import base64
from protocol import JOBS,LIMITS,read,pin,save,verify
from remote import idle,live
import protocol
idle();assert psutil.boot_time()==1789634288.0 and psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(base).free>=LIMITS['preflight_tmpfs']
prior=read(Path({OLDREMOTE!r})/'payload.json');receipt=read(base/'evidence/capture-collection.json')
payload=dict(passed=True,jobs=protocol.JOBS,limits=LIMITS,products={{}},previous_owner=receipt['identities'][0],boot_time=prior['boot_time'],feed=prior['feed'],external=prior['external'],interpreter=prior['interpreter'],files={{p.relative_to(base).as_posix():pin(p) for p in base.rglob('*') if p.is_file()}},scope='Offline complete export of closed traces; no inference.')
save(base/'payload.json',payload);value=verify(base)
result=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',result)
print(json.dumps(dict(**result,payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
'''))
 encoded=result.pop('payload_base64');(BASE/'payload.json').write_bytes(base64.b64decode(encoded));assert pin(BASE/'payload.json')==result['payload'];save(BASE/'staged.json',result);print(json.dumps(result))
def audit():
 previous_closed();capture_run.prepared();assert not (BASE/'closed.json').exists();folder=BASE/'collected'
 receipt=read(folder/'collection.json');state=read(folder/'identity.json')
 assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
 for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
 transfer=read(BASE/'collection-transfer.json');assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(folder/'collection.json')
 assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
 assert [r['name'] for r in state['runs']]==JOBS
 assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
 resources=0;peak=0
 for row in state['runs']:
  assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds'] and all(v==0 for v in row['exitcodes'].values())
  assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
  cpu=0 if row['name'].endswith('-export') else 2
  assert set(row['processes'])=={'worker'} and row['processes']['worker']['affinity']==[cpu]
  samples=[json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()];assert len(samples)==row['samples'] and samples
  for sample in samples:
   check_sample(sample)
   for m in sample['members']:assert m['expected_affinity']==[cpu] and row['members'][str(m['pid'])]==m['birth']
  assert max(s['rss'] for s in samples)==row['peak_rss'];resources+=len(samples);peak=max(peak,row['peak_rss'])
 for name,wanted in read(folder/'built.json')['files'].items():assert pin(folder/name)==wanted,name
 reports={}
 for role in ['current','candidate']:
  path=folder/(role+'-export/events');summary=read(path/'summary.json');old=read(CAPTURE/f'collected/{role}-export/events/summary.json')
  assert summary['complete'] and summary['protocol']=='all-event-records-v1' and summary['lost']==0 and summary['runtime']=='10.0.8'
  run=next(r for r in state['runs'] if r['name']==role+'-export');assert summary['exporter_pid']==run['processes']['worker']['pid']
  assert summary['input_sha256']==old['input_sha256']==pin(folder/(role+'-capture/capture.nettrace'))['sha256']
  events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
  assert len(events)==summary['recorded']==sum(summary['allCounts'].values())
  assert dict(collections.Counter(e['provider']+':'+e['name'] for e in events))==summary['allCounts']==old['allCounts']
  assert sum(e['provider']=='Microsoft-Windows-DotNETRuntime' for e in events)==summary['clr_events']
  assert summary['typedClr']==old['clr_events']
  for i,e in enumerate(events):assert e['index']==i and len(base64.b64decode(e['rawBase64'],validate=True))==e['rawLength'] and e['pointerSize']==8
  keys={'provider','name','id','version','thread','pid','ms','payload'}
  typed=[{k:e[k] for k in keys} for e in events if (e['provider']=='Microsoft-Windows-DotNETRuntime' and e['name'] in old['counts']) or e['provider']=='Lokad-Parakeet-MatMul-Diagnostic']
  previous=[{k:e[k] for k in keys} for e in (json.loads(line) for line in (CAPTURE/f'collected/{role}-export/events/events.jsonl').read_text().splitlines())]
  assert typed==previous
  reports[role]=dict(records=len(events),clr=summary['clr_events'],typedClr=summary['typedClr'],previous_records=len(previous),additional_runtime_records=summary['clr_events']-summary['typedClr'],all_counts=summary['allCounts'])
 save(BASE/'analysis.json',dict(passed=True,inference_executed=False,resources=resources,peak_rss=peak,reports=reports))
 files={p.relative_to(BASE).as_posix():pin(p) for p in folder.rglob('*') if p.is_file()}
 for name in ['analysis.json','prepared.json','payload.json','payload.tar.gz','staged.json','deployment.json','results.tar.gz','collection-transfer.json']:files[name]=pin(BASE/name)
 save(BASE/'closed.json',dict(passed=True,inference_executed=False,files=files));print(json.dumps(dict(closed=pin(BASE/'closed.json'),resources=resources,peak_rss=peak,reports=reports)))
if __name__=='__main__':
 action=sys.argv[1];assert action in ['prepare','stage','launch','observe','collect','audit']
 (globals()[action] if action in ['prepare','stage','audit'] else getattr(capture_run,action))()
