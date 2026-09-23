"""Account all diagnostic clocks, events and stack intervals; never rescore."""
import bisect,collections,csv,json,math
from pathlib import Path
from protocol import pin,read,save
from prepare import ROOT,BASE,previous_closed
FULL=ROOT/'artifacts/parakeet-dispatch-full-export-amd-20260923'
OUT=ROOT/'tests/parakeet/dispatch-events-results'
def csvfile(name,rows):
 with (OUT/name).open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def category(stack,frames):
 text='\n'.join(frames[i]['name'] for i in stack)
 if 'mm_unsafe_vectorized_intrinsics_2x4packed_bump' in text or 'mm_unsafe_vectorized_intrinsics_3x4packed' in text:return 'packed-consumption'
 if 'PackPanelsB' in text:return 'packing'
 if 'mm_unsafe_vectorized_intrinsics(' in text:return 'unpacked-row'
 if 'mm_unsafe_vectorized_intrinsics_2x4' in text:return 'unpacked-consumption'
 return 'other'
def main():
 previous_closed()
 for folder,digest in [(BASE,'c6e1e4d42d3f377f77382a4091ad1150c1a62335a72c7d63a38c728d7a753e19'),(FULL,'35b18e874e1e0c47a7b9e1fe6f2608b0940c1eb431c289c32ba05855cb2b5afa')]:
  assert pin(folder/'closed.json')['sha256']==digest
  for name,wanted in read(folder/'closed.json')['files'].items():assert pin(folder/name)==wanted,name
 assert not OUT.exists() or not any(OUT.iterdir());OUT.mkdir(exist_ok=True)
 analysis=read(BASE/'analysis.json');reports={};allclocks=[];allblocks=[];allloads=[];allpauses=[];allattrs=[];alljit=[];alladdresses=[]
 for role in ['current','candidate']:
  calls=analysis['reports'][role]['intervals'];starts=[c['begin_ms'] for c in calls];ends=[c['end_ms'] for c in calls]
  value=read(BASE/f'collected/{role}-capture/result.json')
  events=[json.loads(line) for line in (FULL/f'collected/{role}-export/events/events.jsonl').read_text().splitlines()]
  def location(at):
   i=bisect.bisect_right(starts,at)-1
   if i>=0 and at<=ends[i]:return (calls[i]['fixture'],calls[i]['iteration'],'inside')
   return (-1,-1,'outside')
  def overlap(start,end):
   i=max(0,bisect.bisect_left(ends,start));found=[]
   while i<len(calls) and starts[i]<end:
    delta=max(0,min(end,ends[i])-max(start,starts[i]))
    if delta:found.append((i,delta))
    i+=1
   return found
  queues=collections.defaultdict(list);compiles=[];loads=[];pending_pauses={};pauses=[]
  for e in events:
   if e['provider']!='Microsoft-Windows-DotNETRuntime':continue
   p=e['payload']
   if e['name']=='Method/JittingStarted':queues[(e['thread'],p['MethodID'])].append(e)
   elif e['name']=='Method/LoadVerbose':
    key=(e['thread'],p['MethodID']);assert queues[key],key;start=queues[key].pop(0)
    compiles.append(dict(method=p['MethodName'],namespace=p['MethodNamespace'],thread=e['thread'],tier=p['OptimizationTier'],start=start['ms'],end=e['ms']))
    if p['MethodNamespace'].startswith('Lokad.Onnx.'):
     fixture,iteration,where=location(e['ms']);row=dict(role=role,ms=e['ms'],fixture=fixture,iteration=iteration,where=where,thread=e['thread'],method=p['MethodName'],namespace=p['MethodNamespace'],tier=p['OptimizationTier'],bytes=int(p['MethodSize']),address=p['MethodStartAddress'])
     allloads.append(row);loads.append(row)
   elif e['name']=='GC/SuspendEEStart':
    assert e['thread'] not in pending_pauses;pending_pauses[e['thread']]=e
   elif e['name']=='GC/RestartEEStop':
    pause=pending_pauses.pop(e['thread'])
    touched=overlap(pause['ms'],e['ms'])
    row=dict(role=role,thread=e['thread'],start_ms=pause['ms'],end_ms=e['ms'],reason=pause['payload']['Reason'],overlap_ms=sum(d for i,d in touched),calls=len(touched))
    pauses.append(row);allpauses.append(row)
  assert not pending_pauses and not any(queues.values())
  for c in compiles:
   touched=overlap(c['start'],c['end']);alljit.append(dict(role=role,**c,overlap_ms=sum(d for i,d in touched),calls=len(touched)))
  gc=[p for p in pauses if p['reason'] in ['SuspendForGC','SuspendForGCPrep']]
  alloc_calls=[]
  for i,(row,c) in enumerate((row,c) for row in value['rows'] for c in row['clocks']):
   interval=calls[i];assert interval['fixture']==row['index'] and interval['iteration']==c['iteration']
   allocation=c['allocatedAfter']-c['allocated'];alloc_calls.append(allocation)
   allclocks.append(dict(role=role,fixture=row['index'],iteration=c['iteration'],warmup=c['warmup'],marker=c['marker'],start=c['start'],stop=c['stop'],ticks=c['ticks'],frequency=value['frequency'],begin_ms=interval['begin_ms'],end_ms=interval['end_ms'],gc0=c['after0']-c['gc0'],gc1=c['after1']-c['gc1'],gc2=c['after2']-c['gc2'],allocated_delta=allocation))
  for row in value['rows']:
   alladdresses.append(dict(role=role,fixture=row['index'],**{operand+'_'+when:row['addresses'+when][operand] for operand in ['a','b','c'] for when in ['Before','After']}))
   for start in range(0,120,10):
    clocks=row['clocks'][start:start+10];allblocks.append(dict(role=role,fixture=row['index'],m=row['m'],reduction=row['reduction'],columns=row['columns'],first=start,last=start+9,warmup=start<60,mean_ms=sum(c['ticks'] for c in clocks)/10/value['frequency']*1000))
  document=read(next((BASE/f'collected/{role}-stacks').glob('*.speedscope.json')));frames=document['shared']['frames'];attrs=collections.defaultdict(float);outside=collections.defaultdict(float);profile_totals={};leaf=collections.defaultdict(float)
  for profile in document['profiles']:
   stack=[];previous=profile['startValue'];total=0.;worker=profile['name']==f"Thread ({value['nativeThread']})"
   for e in profile['events']:
    at=max(previous,e['at']);assert at-e['at']<=.001
    if at>previous:
     delta=at-previous;total+=delta;kind=category(stack,frames) if stack else 'empty';used=0.
     if worker:
      for i,d in overlap(previous,at):
       attrs[(calls[i]['fixture'],calls[i]['iteration'],kind)]+=d;used+=d
       text=frames[stack[-1]]['name'] if stack else '<empty>';leaf[(calls[i]['fixture'],kind,text)]+=d
     outside[(profile['name'],kind)]+=delta-used
    if e['type']=='O':stack.append(e['frame'])
    else:assert e['type']=='C' and stack.pop()==e['frame']
    previous=at
   assert not stack and abs(total-(profile['endValue']-profile['startValue']))<=.001;profile_totals[profile['name']]=total
  assert abs(sum(attrs.values())+sum(outside.values())-sum(profile_totals.values()))<1e-6
  for (fixture,iteration,kind),ms in sorted(attrs.items()):allattrs.append(dict(role=role,fixture=fixture,iteration=iteration,category=kind,estimated_thread_ms=ms))
  aggregated=collections.defaultdict(float)
  for (fixture,iteration,kind),ms in attrs.items():aggregated[(fixture,'warmup' if iteration<60 else 'recorded',kind)]+=ms
  report=dict(records=len(events),compilations=len(compiles),gc_pauses=len(gc),gc_pause_overlap_ms=sum(p['overlap_ms'] for p in gc),gc_calls=sum(p['calls'] for p in gc),allocation_positive_calls=sum(x>0 for x in alloc_calls),total_allocation_delta=sum(alloc_calls),suspension_reasons=dict(collections.Counter(p['reason'] for p in pauses)),stack_profile_ms=profile_totals,stack_inside_ms=sum(attrs.values()),stack_outside_ms=sum(outside.values()),stack_attribution=[dict(fixture=k[0],phase=k[1],category=k[2],estimated_thread_ms=v) for k,v in sorted(aggregated.items())],method_loads=loads)
  reports[role]=report
 csvfile('clocks-20260923.csv',allclocks);csvfile('blocks-20260923.csv',allblocks);csvfile('method-loads-20260923.csv',allloads);csvfile('compilation-20260923.csv',alljit);csvfile('suspensions-20260923.csv',allpauses);csvfile('stack-attribution-20260923.csv',allattrs);csvfile('addresses-20260923.csv',alladdresses)
 save(OUT/'analysis-20260923.json',dict(diagnostic_only=True,no_score=True,capture_closure=pin(BASE/'closed.json'),complete_export_closure=pin(FULL/'closed.json'),reports=reports))
 print(json.dumps({role:{k:v for k,v in report.items() if k not in ['stack_attribution','method_loads']} for role,report in reports.items()},indent=2))
if __name__=='__main__':main()
