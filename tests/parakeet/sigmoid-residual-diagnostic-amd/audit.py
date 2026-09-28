"""Reconcile the unchanged workload and retain unresolved sampled execution."""
import base64
from collections import Counter
import json
import math
import re
import struct
from run import ROOT, TOOLS, BASE, APP, PROFILE, FLAGS, PROVIDERS, load, read, pin, write, prepared

CLR = 'Microsoft-Windows-DotNETRuntime'


def events_checked(events, summary, ready, measured):
    assert summary['complete'] and summary['protocol'] == 'all-event-records-v1'
    assert summary['lost'] == 0 and summary['clr_events'] > 0 and summary['recorded'] == len(events)
    assert [e['index'] for e in events] == list(range(len(events)))
    assert dict(Counter(e['provider']+':'+e['name'] for e in events)) == summary['allCounts']
    assert sum(e['provider'] == CLR for e in events) == summary['clr_events']
    for event in events:
        assert math.isfinite(event['ms']) and event['ms'] >= 0
        assert len(base64.b64decode(event['rawBase64'], validate=True)) == event['rawLength']
    markers = [e for e in events if e['provider'] == 'Lokad-Pyannote-Diagnostic' and e['id'] == 1]
    assert len(markers) == 120 and len(measured) == 60
    assert all(a['ms'] <= b['ms'] for a, b in zip(markers, markers[1:]))
    intervals = []
    for index, row in enumerate(measured):
        pair = markers[index*2:index*2+2]
        for event, phase in zip(pair, ['begin', 'end'], strict=True):
            assert event['pid'] == ready['pid'] and event['thread'] == ready['thread_id']
            assert event['payload'] == dict(phase=phase, name=row['name'], **{'pass':str(row['pass'])})
            raw = base64.b64decode(event['rawBase64'])
            assert struct.unpack('<i', raw[-4:]) == (row['pass'],)
            assert raw[:-4].decode('utf-16-le') == phase+'\0'+row['name']+'\0'
        assert pair[0]['ms'] < pair[1]['ms']
        intervals.append(dict(name=row['name'], pass_index=row['pass'],
            begin_ms=pair[0]['ms'], end_ms=pair[1]['ms'], wall_seconds=row['seconds']))
    pending = None; pauses = []; unmatched = []
    for event in events:
        if event['provider'] != CLR or event['pid'] != ready['pid']:
            continue
        if event['name'] == 'GC/SuspendEEStart':
            if pending is not None:
                unmatched.append(pending)
            pending = event
        elif event['name'] == 'GC/RestartEEStop':
            if pending is None:
                unmatched.append(event)
            else:
                assert pending['ms'] <= event['ms']
                pauses.append(dict(begin_ms=pending['ms'], end_ms=event['ms'],
                    reason=pending['payload'].get('Reason'), begin_index=pending['index'], end_index=event['index']))
                pending = None
    if pending is not None:
        unmatched.append(pending)
    # Do not count overlapping suspensions twice, or assign them to an operator.
    union = []
    for pause in pauses:
        for interval in intervals:
            a, b = max(pause['begin_ms'], interval['begin_ms']), min(pause['end_ms'], interval['end_ms'])
            if a < b:
                if union and a <= union[-1][1]:
                    union[-1][1] = max(union[-1][1], b)
                else:
                    union.append([a, b])
    methods = [e for e in events if e['provider'].startswith(CLR) and
               'Sigmoid' in e['payload'].get('MethodName', '')]
    return dict(request_intervals=intervals, markers=len(markers), events=len(events), lost=summary['lost'],
        suspensions=pauses, unpaired_suspension_events=unmatched, suspension_union_ms=union,
        request_suspension_seconds_per_corpus=sum(b-a for a,b in union)/3000,
        sigmoid_method_events=methods, loaded_version_is_not_execution_proof=True)


def resources(folder, state):
    count = 0; peak = 0
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
        samples = [json.loads(line) for line in (folder/'logs'/(run['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0
        assert max(s['rss'] for s in samples) == run['peak_rss']
        for sample in samples:
            assert sample['seconds'] < 900 and sample['rss'] < 12*1024**3
            assert min(sample['available'], sample['disk']) >= 1024**3
            assert sample['output_bytes'] <= 512*1024**2
            assert sample['rss'] == sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                identity = run['processes'][member['role']]
                assert member['pid'] == identity['pid'] and member['birth'] == identity['birth']
                assert member['affinity'] == identity['affinity'] == ([2] if member['role'] == 'target' else [0])
                assert member['threads'] and all(t['affinity'] == member['affinity'] for t in member['threads'])
        if run['name'] in ['control', 'sampled']:
            assert run['accounting']['valid'] and run['accounting']['foreign_cpu_fraction'] <= .01
            assert run['preflight']['available'] >= 11*1024**3 and run['preflight']['disk'] >= 2*1024**3
            assert run['exit_codes'] == ({'target':0} if run['name'] == 'control' else {'target':0, 'collector':0})
        count += len(samples); peak = max(peak, run['peak_rss'])
    return dict(samples=count, peak_rss=peak)


def main():
    prepared(); assert not (BASE/'closed.json').exists()
    folder = BASE/'collected'; spec = read(BASE/'spec.json')
    receipt = read(folder/'collection.json'); transfer = read(BASE/'transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz')
    assert transfer['collection'] == pin(folder/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0
    for name, wanted in receipt['files'].items():
        assert pin(folder/name) == wanted, name
    assert pin(folder/'spec.json') == pin(BASE/'spec.json')
    state = read(folder/'state.json')
    assert state['complete'] and state['code'] == 0 and state['boot'] == spec['boot']
    assert state['supervisor'] == read(BASE/'deployment.json')
    assert receipt['identities'] == [state['supervisor']]+[i for r in state['runs'] for i in r['processes'].values()]
    assert [r['name'] for r in state['runs']] == ['control', 'sampled', 'speedscope', 'chromium', 'events']
    resource_review = resources(folder, state)
    protocol = load('unchanged_app_protocol', APP/'collected/runtime/protocol.py')
    manifest = read(APP/'collected/manifests/current-parakeet.json')
    previous = read(PROFILE/'capture-collected/control/result.json')
    results = {}; timings = {}
    for run in state['runs'][:2]:
        name = run['name']; output = folder/name
        value = read(output/'result.json'); ready = read(output/'ready.json')
        assert ready == run['ready'] and ready['flags'] == value['flags'] == FLAGS
        assert ready['pid'] == run['processes']['target']['pid']
        assert abs(ready['birth_milliseconds']/1000-run['processes']['target']['birth']) < 1.1
        assert ready['warmup_records'] == 20 and ready['affinity'] == 4 and ready['runtime'] == '10.0.8'
        protocol.validate_records(dict(value, flags={}), manifest, 'timing')
        assert value['passed'] and value['sampled'] == (name == 'sampled') and value['processor_count'] == 1
        for key in ['core_sha256', 'data_sha256', 'runner_sha256', 'manifest_sha256', 'runtime', 'os']:
            assert value[key] == previous[key], key
        assert {p.name for p in output.glob('[0-9][0-9][0-9].json')} == {f'{i:03}.json' for i in range(80)}
        for index, (row, prior) in enumerate(zip(value['records'], previous['records'], strict=True)):
            assert row == read(output/f'{index:03}.json') and row['result'] == prior['result']
            assert row['thread_id'] == ready['thread_id'] and row['cpu_frequency'] == 10000000
            assert min(row['cpu_user_ticks'], row['cpu_system_ticks'], row['allocated_bytes']) >= 0
            assert len(row['gc_before']) == len(row['gc_after']) == 3
            assert all(a <= b for a,b in zip(row['gc_before'], row['gc_after']))
        measured = [r for r in value['records'] if r['phase'] == 'measured']
        timings[name] = dict(corpus_seconds=sum(r['seconds'] for r in measured)/3,
            process_cpu_seconds=sum(r['cpu_user_ticks']+r['cpu_system_ticks'] for r in measured)/3e7,
            allocated_bytes_per_corpus=sum(r['allocated_bytes'] for r in measured)/3,
            collections_per_corpus=[sum(r['gc_after'][g]-r['gc_before'][g] for r in measured)/3 for g in range(3)])
        results[name] = value
        if name == 'sampled':
            command = run['collector_command']
            assert command[command.index('--providers')+1] == PROVIDERS
            assert run['enabled'] == read(output/'collector-enabled.json') and run['enabled']['enabled']
        else:
            assert not (output/'capture.nettrace').exists()
    stacks = load('retained_stacks', TOOLS.parent/'selected-profile-amd/selected_stacks.py')
    speed = read(folder/'exports/speedscope.speedscope.json')
    cross = stacks.cross_export(speed, read(folder/'exports/chromium.chromium.json'))
    parsed = stacks.inspect(speed, {'corpus':'!SampledRequests.FullParakeet('})
    ready = read(folder/'sampled/ready.json')
    assert all(re.match(r'Thread \((\d+)\)', i['thread']).group(1) == str(ready['thread_id']) for i in parsed['intervals']['corpus'])
    frames = speed['shared']['frames']
    assert not any('SampledRequests.WarmupParakeet(' in f['name'] for f in frames)
    process_frames = [f['name'] for f in frames if f['name'].startswith('Process64 dotnet (')]
    assert len(process_frames) == 1 and f"({ready['pid']})" in process_frames[0]
    coverage = parsed['selected_seconds']['corpus']/(timings['sampled']['corpus_seconds']*3)
    assert abs(coverage-1) < .05, coverage
    summary = read(folder/'events/summary.json')
    assert summary['input_sha256'] == pin(folder/'sampled/capture.nettrace')['sha256']
    assert summary['runtime'] == '10.0.8' and summary['exporter_pid'] == state['runs'][-1]['processes']['export']['pid']
    events = [json.loads(line) for line in (folder/'events/events.jsonl').read_text().splitlines()]
    timeline = events_checked(events, summary, ready, results['sampled']['records'][20:])
    selected = {key:[r for r in parsed[key] if 'Sigmoid' in r['method']] for key in ['exclusive', 'inclusive']}
    assert any('SigmoidRationalVector' in r['method'] for r in selected['inclusive'])
    prior_seconds = sum(r['seconds'] for r in previous['records'] if r['phase'] == 'measured')/3
    jit = {}
    for name in ['control', 'sampled']:
        path = folder/'logs'/(name+'-target.log'); text = path.read_text()
        assert 'SigmoidRationalVector' in text and '; Assembly listing for method ' in text
        jit[name] = dict(file=path.relative_to(BASE).as_posix(), **pin(path),
            listings=[line for line in text.splitlines() if line.startswith('; Assembly listing for method ')])
    analysis = dict(passed=True, diagnostic_only=True, product_changed=False, consumer_rebuilt=False,
        requests=160, measured_requests=120, resources=resource_review, timings=timings,
        sampled_to_control=timings['sampled']['corpus_seconds']/timings['control']['corpus_seconds'],
        control_to_previous_uninstrumented=timings['control']['corpus_seconds']/prior_seconds,
        sampled_to_wall=coverage, exports=cross, sigmoid_samples=selected, stacks=parsed, events=timeline,
        jit=jit, instruction_sample_join=False, per_node_sample_join=False,
        limits='Sampled thread-time estimates; native frames may be unresolved. Method load/JIT listings do not prove active tier or instruction cycles. Whole-request GC is not assigned to Sigmoid.')
    write(BASE/'analysis.json', analysis)
    write(BASE/'closed.json', dict(passed=True, diagnostic_only=True, analysis=pin(BASE/'analysis.json'),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(passed=True, closed=pin(BASE/'closed.json'), timings=timings,
        sampled_to_control=analysis['sampled_to_control'], sigmoid=selected,
        request_suspension_seconds_per_corpus=timeline['request_suspension_seconds_per_corpus'])))


if __name__ == '__main__':
    main()
