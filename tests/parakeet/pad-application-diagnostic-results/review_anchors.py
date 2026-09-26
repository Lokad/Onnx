"""Preserve the missing-marker failure; independently reconcile recorded anchors."""
import base64
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT/'tests/parakeet/pad-application-diagnostic-amd'
sys.path.insert(0,str(TOOLS))
import audit as original
from prepare import BASE
from protocol import read, pin, save, JOBS
from scope import compiled
OUT = ROOT/'artifacts/parakeet-padding-anchor-review-20260926'


def reconcile(events, summary, anchors, records, pid):
    assert summary['complete'] and summary['lost'] == 0 and summary['clr_events'] > 0
    assert summary['protocol'] == 'all-event-records-v1' and summary['recorded'] == len(events)
    assert [e['index'] for e in events] == list(range(len(events)))
    assert dict(Counter(e['provider']+':'+e['name'] for e in events)) == summary['allCounts']
    assert sum(e['provider'] == original.CLR for e in events) == summary['clr_events']
    for event in events:
        assert math.isfinite(event['ms']) and event['ms'] >= 0
        assert len(base64.b64decode(event['rawBase64'],validate=True)) == event['rawLength']
    clocks = [e for e in events if e['provider'] == 'Lokad-Parakeet-Clock' and e['id'] == 1]
    missing = [e for e in events if e['provider'] == 'Lokad-Pyannote-Diagnostic' and e['id'] == 1]
    assert not missing, 'This review only covers the documented absence, never a partial marker stream'
    assert len(clocks) == len(anchors['anchors']) == 161 and len(records) == 80
    assert all(e['pid'] == pid for e in clocks)
    assert all(a['ms'] <= b['ms'] for a,b in zip(clocks,clocks[1:]))
    frequency = anchors['frequency'];assert type(frequency) is int and frequency > 0
    assert all(row['frequency'] == frequency for row in records)
    bounds = [];previous = 0
    for event,anchor in zip(clocks,anchors['anchors'],strict=True):
        assert type(anchor['before']) is int and type(anchor['after']) is int
        assert previous < anchor['before'] <= anchor['after'];previous = anchor['after']
        assert event['thread'] == anchor['thread']
        assert struct.unpack('<q',base64.b64decode(event['rawBase64'])) == (anchor['before'],)
        assert event['payload'] == {'counter':str(anchor['before'])}
        bounds.append((event['ms']-anchor['after']*1000/frequency,
                       event['ms']-anchor['before']*1000/frequency))
    lower = max(a for a,b in bounds)-.002;upper = min(b for a,b in bounds)+.002
    assert lower <= upper, 'Clock intervals do not intersect'
    for index,row in enumerate(records):
        before,after = anchors['anchors'][index*2+1:index*2+3]
        assert before['after'] <= row['start_ticks'] < row['end_ticks'] <= after['before']
        assert before['thread'] == after['thread'] == row['thread_id']
        begin,end = clocks[index*2+1:index*2+3]
        assert begin['ms'] < end['ms']
        assert begin['ms'] <= row['start_ticks']*1000/frequency+upper
        assert row['end_ticks']*1000/frequency+lower <= end['ms']
    return dict(lower_ms=lower,upper_ms=upper,uncertainty_ms=upper-lower,
        anchors=len(clocks),paired_request_anchors=160,named_markers=0,
        events=len(events),clr_events=summary['clr_events'],original_marker_gate_passed=False)


def main():
    from run import prepared
    prepared()
    assert not OUT.exists() and not (BASE/'closed.json').exists()
    folder=BASE/'collected';payload=read(BASE/'payload.json');receipt=read(folder/'collection.json')
    state=read(folder/'identity.json');built=read(folder/'built.json');transfer=read(BASE/'collection-transfer.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(folder/'collection.json')
    assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json')
    assert state['boot_time']==1789634288.0 and state['ended']-state['started']<4*3600
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    for name,wanted in payload['files'].items():
        if (folder/name).is_file():assert pin(folder/name)==wanted
    monitored=original.resources(folder,state,payload)
    assert built['passed']
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted
    assert built['exporter']==payload['exporter']==pin(folder/'export-runtime/DispatchEventsExport.dll')
    reference,=[r for r in read(folder/'evidence/observer-instructions.json')['observations'] if r['assembly']=='Lokad.Onnx.Data.dll']
    sys.path.insert(0,str(folder/'tools'))
    from application_protocol import validate_records
    from phase_audit import attribute
    manifest=read(folder/'manifest.json');runs={r['name']:r for r in state['runs']}
    reports={};values={};rejections=[]
    for role in ['current','candidate']:
        inventory=read(folder/(role+'-inventory/instructions.json'))
        review=compiled(inventory,role,payload,built,reference)
        assert read(folder/(role+'-inventory/review.json'))==dict(**review,inventory=pin(folder/(role+'-inventory/instructions.json')))
        path=folder/(role+'-capture/requests');value=read(path/'result.json')
        validate_records(value,manifest,'timing')
        assert value['passed'] and value['sampled'] and value['runtime']=='.NET 10.0.8' and value['processor_count']==1
        assert value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
        assert value['data_sha256']==payload['observer']['sha256'] and value['runner_sha256']==built['consumer']['sha256']
        assert value['manifest_sha256']==pin(folder/'manifest.json')['sha256']
        pid=runs[role+'-capture']['processes']['worker']['pid']
        ready=read(path/'ready.json');startup=read(path/'startup-ready.json');enabled=read(path/'startup-enabled.json')
        assert ready['pid']==startup['pid']==enabled['pid']==pid and ready['warmup_records']==20
        assert ready['thread_id']==startup['thread_id']
        assert startup['counter']<enabled['counter']<value['records'][0]['start_ticks']
        for index,row in enumerate(value['records']):
            assert row==read(path/f'{index:03}.json') and row['thread_id']==ready['thread_id']
        phases=attribute(value,path,'wall')
        event_folder=folder/(role+'-export/events')
        with gzip.open(event_folder/'events.jsonl.gz','rt',encoding='utf8') as stream:events=[json.loads(line) for line in stream]
        summary=read(event_folder/'summary.json')
        assert summary['input_sha256']==pin(folder/(role+'-capture/capture.nettrace'))['sha256']
        anchors=read(path/'clock-anchors.json')
        aligned=reconcile(events,summary,anchors,value['records'],pid)
        # Demonstrate and retain failure of the unchanged original gate.
        try:original.events_checked(events,summary,anchors,value['records'],pid)
        except AssertionError:rejections.append(dict(role=role,gate='160 named request markers',observed=0,required=160))
        else:raise AssertionError('Original marker gate unexpectedly passed')
        reports[role]=dict(alignment=aligned,phases=phases,**original.padding(path,value,events,aligned))
        values[role]=value
    assert [r['result'] for r in values['current']['records']]==[r['result'] for r in values['candidate']['records']]
    assert read(folder/'current-capture/requests/graphs.json')==read(folder/'candidate-capture/requests/graphs.json')
    original_files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=False,preserved_failure=True,admitted=False,diagnostic_only=True,
        reason='The configured legacy provider has zero named request markers in both traces.',
        failed_checks=rejections,all_runtime_jobs_passed=True,complete_public_and_phase_checks_passed=True,
        resources=monitored,requests=160,anchors_recorded=322,events_lost=0,
        terminal_owners=receipt['identities'],original_auditor=pin(TOOLS/'audit.py'),closer=pin(__file__),files=original_files))
    OUT.mkdir()
    analysis=dict(anchor_reconciliation_passed=True,original_protocol_passed=False,admitted=False,diagnostic_only=True,
        original_failure=pin(BASE/'closed.json'),resources=monitored,requests=160,padding_calls=7840,reports=reports,
        limitations=['No named request markers; timing association uses independently recorded scalar anchors.',
            'No uninstrumented control or performance admission.', 'A loaded compiled version is not proof of its execution.'])
    save(OUT/'analysis.json',analysis)
    save(OUT/'closed.json',dict(passed=True,scope='Independent scalar-anchor reconciliation only',
        original_protocol_passed=False,admitted=False,original_failure=pin(BASE/'closed.json'),
        analysis=pin(OUT/'analysis.json'),reviewer=pin(__file__),original_auditor=pin(TOOLS/'audit.py'),
        files={p.relative_to(OUT).as_posix():pin(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps(dict(original_protocol_passed=False,anchor_reconciliation_passed=True,
        resources=monitored,corpus={k:v['corpus'] for k,v in reports.items()},
        alignment={k:v['alignment'] for k,v in reports.items()})))


if __name__=='__main__':main()
