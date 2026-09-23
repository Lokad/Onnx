"""Reconcile diagnostic clocks/events and resources. No performance admission."""
import collections,json,math
from pathlib import Path
from protocol import JOBS,LIMITS,PROVIDERS,check_sample,pin,read,save
from prepare import ROOT,BASE,previous_closed
CUSTOM='Lokad-Parakeet-MatMul-Diagnostic'
CLR='Microsoft-Windows-DotNETRuntime'
def reconcile(value,events,summary,capture):
 assert value['passed'] and value['diagnosticOnly'] and value['protocol']=='parakeet-dispatch-events-v1'
 assert value['calls']==2520 and value['warmups']==value['measured']==1260 and len(value['rows'])==len(capture['entries'])==21
 assert type(value['frequency']) is int and value['frequency']>0
 assert summary['complete'] and summary['lost']==0 and summary['clr_events']>0 and summary['recorded']==len(events)
 assert [e['index'] for e in events]==list(range(len(events)))
 clr=[e for e in events if e['provider']==CLR];markers=[e for e in events if e['provider']==CUSTOM]
 assert len(clr)+len(markers)==len(events) and len(clr)==summary['clr_events']
 assert dict(collections.Counter(e['name'] for e in clr))==summary['counts']
 recorded_counts=collections.Counter(e['provider']+':'+e['name'] for e in events)
 assert all(summary['allCounts'].get(k)==v for k,v in recorded_counts.items())
 assert len(markers)==5040
 assert all(e['pid']==value['pid'] and e['thread']==value['nativeThread'] and e['id'] in [1,2] for e in markers)
 assert all(math.isfinite(e['ms']) and e['ms']>=0 for e in events)
 assert all(a['ms']<=b['ms'] for a,b in zip(markers,markers[1:]))
 prior=0;calls=[];position=0
 for i,(row,fixture) in enumerate(zip(value['rows'],capture['entries'],strict=True)):
  assert row['index']==i and [row[k] for k in ['name','node','m','reduction','columns']]==[fixture[k] for k in ['name','node','m','k','n']]
  assert row['exact'] and row['guards'] and row['inputs'] and row['output']==fixture['y']['sha256']
  assert row['preparationTicks']>0 and len(row['clocks'])==120
  for key in ['addressesBefore','addressesAfter']:assert set(row[key])=={'a','b','c'} and all(type(n) is int and n>0 for n in row[key].values())
  for j,clock in enumerate(row['clocks']):
   assert clock['iteration']==j and clock['warmup']==(j<60)
   assert prior<clock['marker']<=clock['start']<clock['stop'] and clock['ticks']==clock['stop']-clock['start'];prior=clock['stop']
   assert all(clock['after'+str(g)]>=clock['gc'+str(g)]>=0 for g in range(3))
   assert clock['allocatedAfter']>=clock['allocated']>=0
   pair=markers[position:position+2];position+=2
   for event,identifier,counter in zip(pair,[1,2],[clock['marker'],clock['stop']],strict=True):
    assert event['id']==identifier
    assert {k:int(v) for k,v in event['payload'].items()}==dict(fixture=i,iteration=j,counter=counter)
   calls.append(dict(fixture=i,iteration=j,begin_ms=pair[0]['ms'],end_ms=pair[1]['ms'],clock_ms=clock['ticks']*1000/value['frequency'],gc_delta=[clock['after'+str(g)]-clock['gc'+str(g)] for g in range(3)],allocated_delta=clock['allocatedAfter']-clock['allocated']))
 return dict(calls=len(calls),markers=len(markers),clr_events=len(clr),gc_calls=sum(any(c['gc_delta']) for c in calls),intervals=calls)
def stack_inventory(path):
 value=read(path);frames=value['shared']['frames'];profiles=value['profiles'];assert profiles and frames
 inventory=[]
 for p in profiles:
  assert p['type']=='sampled' and p['unit'] in ['milliseconds','seconds','microseconds','nanoseconds']
  samples=p['samples'];weights=p.get('weights',[1]*len(samples));assert len(weights)==len(samples)
  assert all(math.isfinite(w) and w>=0 for w in weights)
  assert all(type(i) is int and 0<=i<len(frames) for s in samples for i in s)
  inventory.append(dict(name=p['name'],unit=p['unit'],samples=len(samples),weight=sum(weights),empty_samples=sum(not s for s in samples),empty_weight=sum(w for s,w in zip(samples,weights) if not s),start=p['startValue'],end=p['endValue']))
 return dict(frames=len(frames),profiles=inventory,samples=sum(p['samples'] for p in inventory))
def main():
 previous_closed();assert not (BASE/'closed.json').exists();prepared=read(BASE/'prepared.json')
 for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
 assert prepared['archive']==pin(BASE/'payload.tar.gz') and prepared['stage']==pin(BASE/'bundle/stage.json')
 folder=BASE/'collected';receipt=read(folder/'collection.json');state=read(folder/'identity.json');payload=read(BASE/'payload.json')
 assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
 transfer=read(BASE/'collection-transfer.json');assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(folder/'collection.json')
 for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
 assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0 and state['supervisor']==read(BASE/'deployment.json')
 assert state['ended']-state['started']<4*3600 and [r['name'] for r in state['runs']]==payload['jobs']==JOBS
 assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
 resources=0;peak=0;runs={r['name']:r for r in state['runs']}
 for row in state['runs']:
  assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds'] and all(c==0 for c in row['exitcodes'].values())
  assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
  worker_cpu=0 if row['name'].endswith(('-export','-stacks')) or row['name']=='tracer-version' else 2
  assert row['processes']['worker']['affinity']==[worker_cpu]
  if row['name'].endswith('-capture'):
   assert set(row['processes'])=={'worker','collector'} and row['processes']['collector']['affinity']==[0]
   command=row['commands']['collector'];assert command[command.index('--providers')+1]==PROVIDERS
   assert int(command[command.index('--process-id')+1])==row['processes']['worker']['pid']
  else:assert set(row['processes'])=={'worker'}
  for identity in row['processes'].values():assert row['members'][str(identity['pid'])]==identity['birth'] and row['affinities'][str(identity['pid'])]==identity['affinity']
  samples=[json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()];assert len(samples)==row['samples'] and samples
  for sample in samples:
   check_sample(sample)
   for member in sample['members']:
    assert row['members'][str(member['pid'])]==member['birth'] and row['affinities'][str(member['pid'])]==member['expected_affinity']
    if row['name'].endswith('-capture'):assert member['pid'] in [v['pid'] for v in row['processes'].values()]
    else:assert member['expected_affinity']==[worker_cpu]
  assert max(s['rss'] for s in samples)==row['peak_rss'];resources+=len(samples);peak=max(peak,row['peak_rss'])
 built=read(folder/'built.json');assert built['passed']
 for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
 for role,files in payload['products'].items():
  for name,wanted in files.items():assert pin(folder/'runtimes'/role/name)==wanted,name
 reports={};capture=read(BASE/'bundle/fixtures/result.json')
 for sequence,role in enumerate(['current','candidate']):
  run=runs[role+'-capture'];value=read(folder/(role+'-capture/result.json'));ready=read(folder/(role+'-capture/ready.json'));enabled=read(folder/(role+'-capture/collector-enabled.json'))
  assert value['sequence']==sequence and value['role']==role and value['runtime']=='10.0.8' and value['flags']=={}
  assert value['pid']==run['processes']['worker']['pid']==ready['pid']==enabled['pid'] and value['nativeThread']==ready['native_thread']
  assert ready['counter']<enabled['counter']<value['rows'][0]['clocks'][0]['marker']
  assert value['assembly']==built['consumer']['sha256'] and value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
  summary=read(folder/(role+'-export/events/summary.json'));assert summary['runtime']=='10.0.8' and summary['exporter_pid']==runs[role+'-export']['processes']['worker']['pid']
  assert summary['input_sha256']==pin(folder/(role+'-capture/capture.nettrace'))['sha256']
  events=[json.loads(line) for line in (folder/(role+'-export/events/events.jsonl')).read_text().splitlines()]
  report=reconcile(value,events,summary,capture);stackfiles=list((folder/(role+'-stacks')).glob('*.speedscope.json'));assert len(stackfiles)==1
  report['stacks']=stack_inventory(stackfiles[0]);assert report['stacks']['samples']>0
  assert any(k.startswith('Microsoft-DotNETCore-SampleProfiler:') and n>0 for k,n in summary['allCounts'].items())
  reports[role]=report
 analysis=dict(passed=True,diagnostic_only=True,root_product_changed=False,resources=resources,peak_rss=peak,products=payload['products'],consumer=built['consumer'],exporter=built['exporter'],reports=reports)
 save(BASE/'analysis.json',analysis)
 files={p.relative_to(BASE).as_posix():pin(p) for p in folder.rglob('*') if p.is_file()}
 for name in ['analysis.json','prepared.json','staged.json','extracted.json','payload.json','deployment.json','collection-transfer.json','results.tar.gz','payload.tar.gz']:files[name]=pin(BASE/name)
 save(BASE/'closed.json',dict(passed=True,diagnostic_only=True,files=files,root_product_changed=False))
 print(json.dumps(dict(closed=pin(BASE/'closed.json'),resources=resources,peak_rss=peak,reports={k:{n:v for n,v in r.items() if n!='intervals'} for k,r in reports.items()})))
if __name__=='__main__':main()
