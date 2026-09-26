"""Reconcile real padding calls and compilation history; never admit a product."""
import base64
from collections import Counter
import gzip
import json
import math
import struct
import sys
from protocol import JOBS, LIMITS, PROVIDERS, pin, read, save, check_sample
from prepare import BASE, ROOT
from scope import compiled

CLR = 'Microsoft-Windows-DotNETRuntime'


def events_checked(events, summary, anchors, records, pid):
    assert summary['complete'] and summary['lost'] == 0 and summary['clr_events'] > 0
    assert summary['protocol'] == 'all-event-records-v1' and summary['recorded'] == len(events)
    assert [e['index'] for e in events] == list(range(len(events)))
    assert dict(Counter(e['provider']+':'+e['name'] for e in events)) == summary['allCounts']
    assert sum(e['provider'] == CLR for e in events) == summary['clr_events']
    for event in events:
        assert math.isfinite(event['ms']) and event['ms'] >= 0
        assert len(base64.b64decode(event['rawBase64'], validate=True)) == event['rawLength']
    clocks = [e for e in events if e['provider'] == 'Lokad-Parakeet-Clock' and e['id'] == 1]
    markers = [e for e in events if e['provider'] == 'Lokad-Pyannote-Diagnostic' and e['id'] == 1]
    assert len(clocks) == len(anchors['anchors']) == 161 and len(markers) == len(records)*2 == 160
    assert all(e['pid'] == pid for e in clocks+markers)
    assert all(a['ms'] <= b['ms'] for a,b in zip(clocks, clocks[1:]))
    frequency = anchors['frequency']; assert frequency == records[0]['frequency']
    bounds = []
    for event, anchor in zip(clocks, anchors['anchors'], strict=True):
        assert anchor['before'] <= anchor['after'] and event['thread'] == anchor['thread']
        assert struct.unpack('<q', base64.b64decode(event['rawBase64'])) == (anchor['before'],)
        assert event['payload'] == {'counter':str(anchor['before'])}
        bounds.append((event['ms']-anchor['after']*1000/frequency,
                       event['ms']-anchor['before']*1000/frequency))
    # Two microseconds allow serialized EventPipe/float timestamp rounding.
    # Record the complete conservative interval; never pick an offset to align
    # a compilation event with a particular Pad invocation.
    lower = max(a for a,b in bounds)-.002
    upper = min(b for a,b in bounds)+.002
    assert lower <= upper, 'Clock anchor intervals do not intersect'
    for index, row in enumerate(records):
        begin, end = markers[index*2:index*2+2]
        assert begin['ms'] < end['ms']
        for event, phase in [(begin, 'begin'), (end, 'end')]:
            assert event['thread'] == row['thread_id']
            assert event['payload'] == {'phase':phase, 'name':row['name'], 'pass':str(row['pass'])}
            raw = base64.b64decode(event['rawBase64'])
            assert struct.unpack('<i', raw[-4:]) == (row['pass'],)
            assert raw[:-4].decode('utf-16-le') == phase+'\0'+row['name']+'\0'
        before, after = anchors['anchors'][index*2+1:index*2+3]
        assert before['after'] <= row['start_ticks'] < row['end_ticks'] <= after['before']
        assert clocks[index*2+1]['ms'] <= begin['ms'] < end['ms'] <= clocks[index*2+2]['ms']
    return dict(lower_ms=lower, upper_ms=upper, uncertainty_ms=upper-lower,
                anchors=len(clocks), markers=len(markers), events=len(events), clr_events=summary['clr_events'])


def padding(folder, result, events, alignment):
    metadata = read(folder/'graphs.json')
    nodes = {g:{n['id']:n for n in d['nodes']} for g,d in metadata.items()}
    expected = {'nemo128.onnx':1, 'encoder-model.onnx':48, 'decoder_joint-model.onnx':0}
    assert {g:sum(n['op'] == 'Pad' for n in d.values()) for g,d in nodes.items()} == expected
    method_events = [e for e in events if e['provider'] == CLR and
        e['name'] in ['Method/JittingStarted', 'Method/LoadVerbose', 'Method/UnloadVerbose'] and
        e['payload'].get('MethodNamespace') == 'Lokad.Onnx.CPUExecutionProvider' and
        e['payload'].get('MethodName') in ['PadCore', 'PadDispatch', 'Pad', 'PadReflectCore']]
    assert any(e['payload'].get('MethodName') == 'PadCore' and e['name'] == 'Method/LoadVerbose' for e in method_events)
    suspensions = []; pending = None; unmatched = []
    for event in events:
        if event['provider'] != CLR: continue
        if event['name'] == 'GC/SuspendEEStart':
            if pending is not None: unmatched.append(pending)
            pending = event
        elif event['name'] == 'GC/RestartEEStop':
            if pending is None: unmatched.append(event)
            else:
                assert pending['ms'] <= event['ms']
                suspensions.append(dict(begin_index=pending['index'],end_index=event['index'],
                    begin_ms=pending['ms'],end_ms=event['ms'],reason=pending['payload'].get('Reason')))
                pending = None
    if pending is not None: unmatched.append(pending)
    requests = []; calls = []
    for index, row in enumerate(result['records']):
        value = read(folder/f'phase-{index:03}.json'); current = []
        for call in value['calls']:
            graph = call['graph']
            for node in call['nodes']:
                descriptor = nodes[graph][node['NodeId']]
                if descriptor['op'] != 'Pad': continue
                a = node['StartTicks']*1000/row['frequency']; b = node['EndTicks']*1000/row['frequency']
                start_low, start_high = a+alignment['lower_ms'], a+alignment['upper_ms']
                end_low, end_high = b+alignment['lower_ms'], b+alignment['upper_ms']
                current.append(dict(request=index, name=row['name'], pass_index=row['pass'], phase=row['phase'],
                    graph=graph, node=descriptor, start_ticks=node['StartTicks'], end_ticks=node['EndTicks'],
                    seconds=(node['EndTicks']-node['StartTicks'])/row['frequency'],
                    start_ms_bounds=[start_low,start_high], end_ms_bounds=[end_low,end_high],
                    method_events_possibly_overlapping=[e['index'] for e in method_events if start_low <= e['ms'] <= end_high],
                    suspensions_possibly_overlapping=[s for s in suspensions if s['begin_ms'] <= end_high and s['end_ms'] >= start_low],
                    methods_loaded_before=[e['index'] for e in method_events if e['name'] == 'Method/LoadVerbose' and e['ms'] < start_low]))
        assert len(current) == 49 and current[0]['graph'] == 'nemo128.onnx'
        assert all(c['graph'] == 'encoder-model.onnx' for c in current[1:])
        requests.append(dict(index=index,name=row['name'],pass_index=row['pass'],phase=row['phase'],
            seconds=row['seconds'],frontend_padding_seconds=current[0]['seconds'],
            encoder_padding_seconds=sum(c['seconds'] for c in current[1:]),
            allocated_bytes=row['allocated_bytes'],gc_before=row['gc_before'],gc_after=row['gc_after']))
        calls.extend(current)
    assert len(calls) == 3920
    return dict(requests=requests, calls=calls, compilation_events=method_events,
        suspensions=suspensions,unpaired_suspension_events=unmatched,
        loaded_version_is_not_execution_proof=True,
        corpus={key:sum(r[key] for r in requests if r['phase'] == 'measured')/3
                for key in ['seconds','frontend_padding_seconds','encoder_padding_seconds']})


def resources(folder, state, payload):
    total = 0; peak = 0
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert all(c == 0 for c in row['exitcodes'].values())
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        cpu = 0 if row['name'].endswith('-export') or row['name'] == 'tracer-version' else 2
        assert row['processes']['worker']['affinity'] == [cpu]
        if row['name'].endswith('-capture'):
            assert set(row['processes']) == {'worker','collector'} and row['processes']['collector']['affinity'] == [0]
            cmd = row['commands']['collector']
            assert cmd[cmd.index('--providers')+1] == PROVIDERS
            assert int(cmd[cmd.index('--process-id')+1]) == row['processes']['worker']['pid']
        else: assert set(row['processes']) == {'worker'}
        samples = [json.loads(s) for s in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert samples and len(samples) == row['samples'] and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert row['members'][str(member['pid'])] == member['birth']
                assert row['affinities'][str(member['pid'])] == member['expected_affinity']
        gaps = [samples[0]['seconds']] + [b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])] + [row['seconds']-samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        total += len(samples); peak = max(peak,row['peak_rss'])
    return dict(samples=total,peak_rss=peak)


def main():
    from run import prepared
    prepared()
    assert not (BASE/'closed.json').exists()
    folder = BASE/'collected'; payload = read(BASE/'payload.json')
    receipt = read(folder/'collection.json'); state = read(folder/'identity.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert receipt['payload'] == pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(folder/'collection.json')
    assert state['complete'] and state['code'] == 0 and state['supervisor'] == read(BASE/'deployment.json')
    assert state['boot_time'] == 1789634288.0 and state['ended']-state['started'] < 4*3600
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    for name,wanted in payload['files'].items():
        if (folder/name).is_file(): assert pin(folder/name) == wanted
    monitored = resources(folder,state,payload)
    built = read(folder/'built.json'); assert built['passed']
    for name,wanted in built['files'].items(): assert pin(folder/name) == wanted
    assert built['exporter'] == payload['exporter'] == pin(folder/'export-runtime/DispatchEventsExport.dll')
    original, = [r for r in read(folder/'evidence/observer-instructions.json')['observations'] if r['assembly'] == 'Lokad.Onnx.Data.dll']
    sys.path.insert(0,str(folder/'tools'))
    from application_protocol import validate_records
    from phase_audit import attribute
    manifest = read(folder/'manifest.json')
    runs = {r['name']:r for r in state['runs']}; reports = {}; values = {}
    for role in ['current','candidate']:
        inventory = read(folder/(role+'-inventory/instructions.json'))
        review = compiled(inventory,role,payload,built,original)
        assert read(folder/(role+'-inventory/review.json')) == dict(**review,inventory=pin(folder/(role+'-inventory/instructions.json')))
        path = folder/(role+'-capture/requests'); value = read(path/'result.json')
        validate_records(value,manifest,'timing')
        assert value['passed'] and value['sampled'] and value['runtime'] == '.NET 10.0.8' and value['processor_count'] == 1
        assert value['core_sha256'] == payload['products'][role]['Lokad.Onnx.dll']['sha256']
        assert value['data_sha256'] == payload['observer']['sha256'] and value['runner_sha256'] == built['consumer']['sha256']
        assert value['manifest_sha256'] == pin(folder/'manifest.json')['sha256']
        pid = runs[role+'-capture']['processes']['worker']['pid']
        ready = read(path/'ready.json'); startup = read(path/'startup-ready.json'); enabled = read(path/'startup-enabled.json')
        assert ready['pid'] == startup['pid'] == enabled['pid'] == pid and ready['warmup_records'] == 20
        assert ready['thread_id'] == startup['thread_id']
        assert startup['counter'] < enabled['counter'] < value['records'][0]['start_ticks']
        for index,row in enumerate(value['records']):
            assert row == read(path/f'{index:03}.json') and row['thread_id'] == ready['thread_id']
        phase = attribute(value,path,'wall')
        event_folder = folder/(role+'-export/events')
        with gzip.open(event_folder/'events.jsonl.gz','rt',encoding='utf8') as stream:
            events = [json.loads(line) for line in stream]
        summary = read(event_folder/'summary.json')
        assert summary['input_sha256'] == pin(folder/(role+'-capture/capture.nettrace'))['sha256']
        aligned = events_checked(events,summary,read(path/'clock-anchors.json'),value['records'],pid)
        report = padding(path,value,events,aligned)
        reports[role] = dict(alignment=aligned,phases=phase,**report)
        values[role] = value
    assert [r['result'] for r in values['current']['records']] == [r['result'] for r in values['candidate']['records']]
    assert read(folder/'current-capture/requests/graphs.json') == read(folder/'candidate-capture/requests/graphs.json')
    analysis = dict(passed=True,diagnostic_only=True,admitted=False,rejected_screen_preserved=True,
        products=payload['products'],observer=payload['observer'],consumer=built['consumer'],
        resources=monitored,requests=160,padding_calls=7840,reports=reports)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,diagnostic_only=True,admitted=False,terminal_owners=receipt['identities'],
        analysis=pin(BASE/'analysis.json'),auditor=pin(__file__),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(passed=True,resources=monitored,requests=160,padding_calls=7840,
        observations={k:dict(corpus=v['corpus'],alignment=v['alignment'],compilations=v['compilation_events']) for k,v in reports.items()})))


if __name__ == '__main__': main()
