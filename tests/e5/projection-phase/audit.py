"""Audit the fixed phase diagnostic and correlate exact call intervals to runtime events."""
from pathlib import Path
import argparse,collections,copy,hashlib,json,math,statistics,tempfile

base=Path()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_files(directory,pins):
    for rel,pin in pins.items():
        path=directory/rel
        assert sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],rel

def correlate(result,events,capture):
    assert capture['completed'] and capture['lost']==0 and capture['pid']==result['process_id']
    assert len(events)==capture['events'] and all(e['pid']==result['process_id'] for e in events)
    markers=[e for e in events if e['provider']=='Lokad-Packed-Phase']
    assert len(markers)==capture['markers']==2*(len(result['intervals'])+1)
    indexed={}
    for e in markers:
        key=int(e['payload']['id']),int(e['payload']['edge'])
        assert key not in indexed and key[1] in (0,1)
        indexed[key]=e
    assert (-1,0) in indexed and (-1,1) in indexed
    frequency=result['stopwatch_frequency'];scale=1000/frequency
    lower=[];upper=[]
    for row in result['intervals']:
        for edge,prefix in ((0,'MarkerBegin'),(1,'MarkerEnd')):
            e=indexed[(row['Id'],edge)]
            assert int(e['payload']['ticks'])==row[prefix+'Before']
            lower.append(e['ms']-row[prefix+'After']*scale)
            upper.append(e['ms']-row[prefix+'Before']*scale)
    lo,hi=max(lower),min(upper)
    # Allow 100 ns of timestamp conversion/rounding uncertainty on each side,
    # without broadening any output or performance gate.
    assert lo-hi<=.0002,('Clock intervals do not intersect',lo-hi)
    offset=(lo+hi)/2;uncertainty=max(0,hi-lo)/2+.0001
    open_jit={};jit=[];unmatched_load=[];open_background={};background=[];unmatched_background=[]
    for e in events:
        p=e['payload'];name=e['name']
        if name=='Method/JittingStarted':
            key=e['thread'],p['MethodID'];assert key not in open_jit,('Duplicate JIT start',key)
            open_jit[key]=e
        elif name=='Method/LoadVerbose':
            key=e['thread'],p['MethodID'];start=open_jit.pop(key,None)
            if start is None:unmatched_load.append(e);continue
            assert start['ms']<=e['ms'] and start['payload']['MethodName']==p['MethodName']
            jit.append(dict(start=start['ms'],end=e['ms'],thread=e['thread'],method_id=p['MethodID'],method=p['MethodNamespace']+':'+p['MethodName'],tier=e['tier'],size=int(p['MethodSize'])))
        elif 'TieredCompilation/BackgroundJitStart' in name:
            assert e['thread'] not in open_background
            open_background[e['thread']]=e
        elif 'TieredCompilation/BackgroundJitStop' in name:
            start=open_background.pop(e['thread'],None)
            if start is None:unmatched_background.append(e);continue
            background.append(dict(start=start['ms'],end=e['ms'],thread=e['thread'],payload=p))
    assert len([e for e in events if e['name']=='Method/JittingStarted'])==capture['methods']
    correlated=[]
    for row in result['intervals']:
        start=row['Start']*scale+offset;end=row['End']*scale+offset
        overlaps=[]
        for event in jit:
            overlap=max(0,min(end,event['end'])-max(start,event['start']))
            if overlap>0:overlaps.append(dict(**event,overlap_ms=overlap))
        bg=[event for event in background if event['end']>start and event['start']<end]
        gc=[event for event in events if event['name'].startswith('GC/') and start<=event['ms']<=end]
        correlated.append(dict(id=row['Id'],trace_start=start,trace_end=end,jit=overlaps,background=bg,gc=gc))
    return dict(clock_offset_ms=offset,clock_uncertainty_ms=uncertainty,jit=jit,background=background,unmatched_jit_starts=list(open_jit.values()),unmatched_loads=unmatched_load,unmatched_background=unmatched_background+list(open_background.values()),intervals=correlated)

def validate(result,order,full,traced,producer_sha256):
    assert result['checked_cases']==256 and len(result['checks'])==256 and result['refusals']==8
    assert result['order']==order and result['processor_count']==1 and result['affinity']==4 and result['avx512']
    assert result['runtime']=='10.0.8'
    assert result['trace_enabled']==traced and result['flags']==({'DOTNET_TieredCompilation':'0'} if full else {})
    assert result['core_sha256']=='7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9'
    assert result['producer_sha256']==producer_sha256
    assert len(result['results'])==60 and len(result['bank'])==20 and len(result['intervals'])==640
    assert [r['Id'] for r in result['intervals']]==list(range(640))
    ids=set();previous=0;by_key={}
    for row in result['intervals']:
        assert row['MarkerBeginBefore']<=row['MarkerBeginAfter']<=row['Start']<row['End']<=row['MarkerEndBefore']<=row['MarkerEndAfter']
        assert row['MarkerBeginBefore']>=previous;previous=row['MarkerEndAfter']
        assert row['ProcessCpuAfter']>=row['ProcessCpuBefore'] and row['ThreadCpuAfter']>=row['ThreadCpuBefore']
        key=row['Region'],row['M'],row['N'],row['K'],row['Mode'],row['Sample']
        assert key not in ids;ids.add(key);by_key[key]=row
    for region in ('isolated','bank'):
        records=result['results' if region=='isolated' else 'bank']
        for record in records:
            m,n,k=record['m'],record.get('n',0),record.get('k',0);mode=record['mode'];shape=record['shape']
            sequence=[(i+order%5+shape)%5 for i in range(5)]
            if order>=5:sequence.reverse()
            assert sequence==record['sequence']
            assert len(record['samples_ms'])==7 and all(math.isfinite(v) and v>0 for v in record['samples_ms'])
            if region=='bank':assert record['matrices']==72 and record['packed_bytes']==84934656
            iterations=record['iterations'] if region=='isolated' else 1
            warm=by_key[region,m,n,k,mode,-1]
            assert warm['Phase']=='warmup' and warm['Iterations']==(8 if region=='isolated' else 2)
            for i,value in enumerate(record['samples_ms']):
                row=by_key[region,m,n,k,mode,i]
                assert row['Phase']=='measured' and row['Iterations']==iterations
                measured=(row['End']-row['Start'])*1000/result['stopwatch_frequency']/iterations
                assert math.isclose(value,measured,rel_tol=1e-13)
    return by_key

def main():
    global base
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args();base=args.artifact.resolve()
    assert not (base/'summary.json').exists() and not (base/'correlation.json').exists()
    archive=read(base/'collected.json');assert archive['complete']=='0' and sha(base/'results.tar.gz')==archive['sha256']
    root=base/'collected/result';identity=read(root/'identity.json');assert identity['complete'] and len(identity['runs'])==8
    verify_files(base,read(base/'bundle.json')['files'])
    summary=[];all_jit=[];first_checks=None;by_condition={};refusal_example=None
    conditions=[('control-off',False,False),('control-on',False,True),('fixed-on',True,True),('fixed-off',True,False)] if args.confirmation else [('normal-off',False,False),('normal-on',False,True),('full-on',True,True),('full-off',True,False)]
    expected=[(order,*c) for order in (0,5) for c in (conditions if order==0 else reversed(conditions))]
    for index,(job,(order,condition,full,traced)) in enumerate(zip(identity['runs'],expected)):
        assert (job['index'],job['order'],job['condition'],job['traced'])==(index,order,condition,traced)
        assert job['target_code']==0 and (job['collector_code']==0 if traced else job['collector_code'] is None)
        assert job['target_members'] and all(v['affinity']=='2' for v in job['target_members'].values())
        assert all(v['affinity']=='0' for v in job['collector_members'].values()) and bool(job['collector_members'])==traced
        directory=root/job['name'];result=read(directory/'samples.json');assert result['process_id']==job['pid']
        producer=base/('control-bin' if args.confirmation and not full else 'probe-bin')/'PhaseProbe.dll'
        mapping=validate(result,order,full and not args.confirmation,traced,sha(producer))
        checks=[{k:r[k] for k in ('m','n','k','exceptional','a_sha256','b_sha256','output_sha256','accumulated_sha256','twice_sha256')} for r in result['checks']]
        if first_checks is None:first_checks=checks
        else:assert checks==first_checks
        correlation=None
        if traced:
            events=[json.loads(line) for line in (directory/'trace/events.jsonl').read_text(encoding='utf-8').splitlines()]
            capture=read(directory/'trace/result.json');correlation=correlate(result,events,capture)
            all_jit.append(dict(job=job['name'],**correlation))
            if refusal_example is None:refusal_example=(result,events,capture)
        bank=[]
        for record in result['bank']:
            m,mode=record['m'],record['mode'];rows=[mapping['bank',m,0,0,mode,i] for i in range(7)]
            detail=[]
            for row in rows:
                wall=(row['End']-row['Start'])*1000/result['stopwatch_frequency']
                item=dict(id=row['Id'],wall_ms=wall,process_cpu_ms=(row['ProcessCpuAfter']-row['ProcessCpuBefore'])/1e6,thread_cpu_ms=(row['ThreadCpuAfter']-row['ThreadCpuBefore'])/1e6)
                if correlation:item['events']=correlation['intervals'][row['Id']]
                detail.append(item)
            bank.append(dict(m=m,mode=mode,position=record['sequence'].index(mode),mean_ms=statistics.mean(record['samples_ms']),samples=detail,slow=m==30 and max(record['samples_ms'])>15,gc_delta=[b-a for a,b in zip(record['gc_before'],record['gc_after'])]))
        item=dict(name=job['name'],order=order,condition=condition,flags=job['flags'],foreign_cpu_fraction=job['accounting']['foreign_cpu_fraction'],bank=bank)
        if correlation:item['clock_uncertainty_ms']=correlation['clock_uncertainty_ms']
        summary.append(item)
        print(job['name'],[(b['mode'],b['position'],round(b['mean_ms'],3),b['slow']) for b in bank if b['m']==30])
    refusals(*refusal_example)
    with (base/'summary.json').open('x',encoding='utf-8') as stream:json.dump(dict(workers=summary,archive_sha256=archive['sha256'],bundle_sha256=sha(base/'bundle.tar.gz'),refusals=8),stream,indent=2)
    with (base/'correlation.json').open('x',encoding='utf-8') as stream:json.dump(all_jit,stream,indent=2)

def refusals(result,events,capture):
    for kind in range(7):
        r,e,c=copy.deepcopy(result),copy.deepcopy(events),copy.deepcopy(capture)
        if kind==0:c['lost']=1
        elif kind==1:e.pop()
        elif kind==2:
            marker=next(x for x in e if x['provider']=='Lokad-Packed-Phase' and x['payload']['id']=='0');marker['payload']['ticks']='0'
        elif kind==3:
            marker=next(x for x in e if x['provider']=='Lokad-Packed-Phase' and x['payload']['id']=='0');marker['ms']+=100
        elif kind==4:c['pid']+=1
        elif kind==5:r['stopwatch_frequency']*=2
        else:
            first=next(x for x in e if x['name']=='Method/JittingStarted')
            load=next(x for x in e if x['name']=='Method/LoadVerbose' and x['thread']==first['thread'] and x['payload']['MethodID']==first['payload']['MethodID'])
            load['payload']['MethodName']='WrongMethod'
        try:correlate(r,e,c)
        except (AssertionError,KeyError):continue
        raise AssertionError('Damaged trace accepted')
    with tempfile.TemporaryDirectory(prefix='onnx-phase-refusal-') as folder:
        directory=Path(folder);path=directory/'binary.dll';path.write_bytes(b'original')
        pin={'binary.dll':dict(bytes=8,sha256=sha(path))};verify_files(directory,pin)
        path.write_bytes(b'modified')
        try:verify_files(directory,pin)
        except AssertionError:return
        raise AssertionError('Changed binary accepted')

if __name__=='__main__':main()
