"""Close diagnostic evidence without a speed or release-admission score."""
import base64,collections,csv,gzip,hashlib,json,math,struct
from pathlib import Path
from protocol import JOBS,LIMITS,PROVIDERS,ROLES,KEYS,CALLS,BLOCK,check_sample,pin,read,save
from checks import compiled_scope,observer_scope,merge_observation
from source_scope import verify as verify_source
from prepare import ROOT,BASE,previous_closed
CUSTOM='Lokad-Parakeet-MatMul-Diagnostic'
CLR='Microsoft-Windows-DotNETRuntime'

def reconcile(value,events,summary):
    assert value['passed'] and value['diagnosticOnly'] and value['mode']=='timing'
    assert value['calls']==len(value['clocks'])==CALLS
    assert summary['complete'] and summary['lost']==0 and summary['clr_events']>0
    assert summary['protocol']=='all-event-records-v1' and summary['recorded']==len(events)
    assert [e['index'] for e in events]==list(range(len(events)))
    assert dict(collections.Counter(e['provider']+':'+e['name'] for e in events))==summary['allCounts']
    assert sum(e['provider']==CLR for e in events)==summary['clr_events']
    assert all(math.isfinite(e['ms']) and e['ms']>=0 for e in events)
    for event in events:assert len(base64.b64decode(event['rawBase64']))==event['rawLength']
    markers=[e for e in events if e['provider']==CUSTOM and e['id'] in [1,2]]
    assert len(markers)==CALLS*2
    assert all(e['pid']==value['pid'] and e['thread']==value['nativeThread'] for e in markers)
    assert all(a['ms']<=b['ms'] for a,b in zip(markers,markers[1:]))
    prior=0;calls=[]
    for index,clock in enumerate(value['clocks']):
        assert clock['index']==index and clock['warmup']==(index<600)
        assert prior<clock['marker']<=clock['start']<clock['end'] and clock['ticks']==clock['end']-clock['start'];prior=clock['end']
        assert type(clock['frequency']) is int and clock['frequency']>0
        assert clock['cpuAfter']>=clock['cpuBefore']>=0 and clock['allocatedAfter']>=clock['allocated']>=0
        assert all(clock['after'+str(g)]>=clock['gc'+str(g)]>=0 for g in range(3))
        pair=markers[index*2:index*2+2]
        for event,identifier,counter in zip(pair,[1,2],[clock['marker'],clock['end']],strict=True):
            assert event['id']==identifier
            assert struct.unpack('<iiq',base64.b64decode(event['rawBase64']))==(0,index,counter)
            assert {k:int(v) for k,v in event['payload'].items()}==dict(fixture=0,iteration=index,counter=counter)
        calls.append(dict(index=index,phase=('warmup' if index<600 else 'original-measured-label' if index<780 else 'diagnostic-extension'),begin_ms=pair[0]['ms'],end_ms=pair[1]['ms'],
            wall_ms=clock['ticks']*1000/clock['frequency'],cpu_ms=(clock['cpuAfter']-clock['cpuBefore'])/10000,
            allocated_bytes=clock['allocatedAfter']-clock['allocated'],
            gc0=clock['after0']-clock['gc0'],gc1=clock['after1']-clock['gc1'],gc2=clock['after2']-clock['gc2']))
    blocks=[]
    for start in range(0,CALLS,BLOCK):
        group=calls[start:start+BLOCK]
        blocks.append(dict(first=start,last=start+BLOCK-1,wall_ms=sum(c['wall_ms'] for c in group)/BLOCK,
            cpu_ms=sum(c['cpu_ms'] for c in group)/BLOCK,allocated_bytes=sum(c['allocated_bytes'] for c in group)/BLOCK,
            gc_calls=sum(any(c[k] for k in ['gc0','gc1','gc2']) for c in group)))
    return dict(calls=calls,blocks=blocks,markers=len(markers),events=len(events),clr_events=summary['clr_events'])

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    from run import prepared
    prepared();folder=BASE/'collected';receipt=read(folder/'collection.json');state=read(folder/'identity.json');payload=read(BASE/'payload.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for n,v in receipt['files'].items():assert pin(folder/n)==v,n
    assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS and state['ended']-state['started']<4*3600
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    transfer=read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(folder/'collection.json')
    resources=0;peak=0;runs={r['name']:r for r in state['runs']}
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds'] and all(c==0 for c in row['exitcodes'].values())
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        cpu=0 if row['name'].endswith('-export') or row['name'] in ['tracer-version','exporter-roundtrip'] else 2
        assert row['processes']['worker']['affinity']==[cpu]
        if row['name'].endswith('-capture'):
            assert set(row['processes'])=={'worker','collector'} and row['processes']['collector']['affinity']==[0]
            cmd=row['commands']['collector'];assert cmd[cmd.index('--providers')+1]==PROVIDERS
            assert int(cmd[cmd.index('--process-id')+1])==row['processes']['worker']['pid']
        else:assert set(row['processes'])=={'worker'}
        samples=[json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert samples and len(samples)==row['samples'] and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for m in sample['members']:assert row['members'][str(m['pid'])]==m['birth'] and row['affinities'][str(m['pid'])]==m['expected_affinity']
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    for n,v in built['files'].items():assert pin(folder/n)==v,n
    review=compiled_scope(read(folder/'consumer-inventory/instructions.json'),payload,built)
    assert review==read(folder/'consumer-inventory/review.json')
    assert verify_source((folder/'evidence/OriginalProgram.cs.txt').read_text(),(folder/'source/consumer/Program.cs').read_text())==read(folder/'evidence/source-review.json')
    observer_review=observer_scope(read(folder/'observer-inventory/instructions.json'),payload,built)
    assert observer_review==read(folder/'observer-inventory/review.json')
    assert built['exporter']==payload['reused_exporter']['binary']==pin(folder/'export-runtime/DispatchEventsExport.dll')
    assert payload['diagnostic_only'] and not payload['release_admitted']
    failed=[r for r in read(folder/'evidence/graph-analysis.json')['performance'] if not r['regression_passed']]
    assert failed==payload['failed_release_cases'] and [r['key'] for r in failed]==['e5-8tok']
    reports={}
    for role,product in ROLES.items():
        original=read(folder/f'evidence/original-{KEYS[role]}-{product}.json')
        for n,v in payload['products'][product].items():assert pin(folder/'runtimes'/product/n)==v,n
        value=merge_observation(read(folder/(role+'-capture/output/result.json')),read(folder/(role+'-capture/output/diagnostic.json')))
        ready=read(folder/(role+'-capture/ready.json'));enabled=read(folder/(role+'-capture/collector-enabled.json'))
        assert value['pid']==runs[role+'-capture']['processes']['worker']['pid']==ready['pid']==enabled['pid']
        assert value['nativeThread']==ready['native_thread'] and ready['counter']<enabled['counter']<value['clocks'][0]['marker']
        assert value['key']==KEYS[role]
        assert value['runtime']=='10.0.8' and value['flags']=={} and value['core']==payload['products'][product]['Lokad.Onnx.dll']['sha256'] and value['consumer']==built['consumer']['sha256']
        assert value['arrays']==original['arrays'] and value['inputs_unchanged'] and value['held_outputs_unchanged']
        for row in value['arrays']:assert pin(folder/(role+'-capture/output')/row['file'])['sha256']==row['sha256']
        summary=read(folder/(role+'-export/events/summary.json'))
        assert summary['input_sha256']==pin(folder/(role+'-capture/capture.nettrace'))['sha256']
        assert summary['runtime']=='10.0.8' and summary['exporter_pid']==runs[role+'-export']['processes']['worker']['pid']
        with gzip.open(folder/(role+'-export/events/events.jsonl.gz'),'rt',encoding='utf8') as stream:
            events=[json.loads(line) for line in stream]
        reports[role]=reconcile(value,events,summary)
    assert len(reports)==4 and sum(len(r['calls']) for r in reports.values())==4*CALLS
    assert sum(r['markers'] for r in reports.values())==8*CALLS
    save(BASE/'analysis.json',dict(passed=True,diagnostic_only=True,root_product_changed=False,resources=resources,peak_rss=peak,
        release_admitted=False,failed_release_cases=failed,reused_exporter=payload['reused_exporter'],observer_review=observer_review,keys=KEYS,
        products=payload['products'],consumer=built['consumer'],exporter=built['exporter'],compiled_review=review,reports=reports))
    files={p.relative_to(BASE).as_posix():pin(p) for p in folder.rglob('*') if p.is_file()}
    for n in ['analysis.json','prepared.json','staged.json','payload.json','deployment.json','collection-transfer.json','results.tar.gz','payload.tar.gz']:files[n]=pin(BASE/n)
    save(BASE/'closed.json',dict(passed=True,diagnostic_only=True,root_product_changed=False,release_admitted=False,analysis=pin(BASE/'analysis.json'),files=files))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),resources=resources,peak_rss=peak,
        reports={k:{n:v for n,v in r.items() if n not in ['calls','blocks']} for k,r in reports.items()})))

if __name__=='__main__':main()
