"""Reconcile native phases and ORT events; never publish profiler clocks as benchmarks."""
from bisect import bisect_right
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from run import APP, BASE, ROOT, pin, read, write


def exclusive(events):
    """Subtract nested node intervals once, retaining parent self time."""
    rows = sorted(events, key=lambda e: (e['ts'], -e['dur']))
    stack = []
    totals = []
    for event in rows:
        start, end = event['ts'], event['ts']+event['dur']
        assert event['dur'] >= 0
        while stack and start >= stack[-1][0]:
            stack.pop()
        own = event['dur']
        if stack:
            assert end <= stack[-1][0], 'Partially overlapping node intervals'
            totals[stack[-1][1]][1] -= own
        totals.append([event, own]); stack.append((end, len(totals)-1))
    assert all(duration >= 0 for _, duration in totals)
    return totals


def profile_graph(events, calls, requests):
    runs = sorted([e for e in events if e.get('name') == 'model_run' and e.get('ph') == 'X'], key=lambda e: e['ts'])
    assert len(runs) == len(calls), (len(runs), len(calls))
    assert all(a['ts']+a['dur'] <= b['ts'] for a, b in zip(runs, runs[1:]))
    starts = [e['ts'] for e in runs]
    groups = defaultdict(list)
    for e in events:
        if e.get('cat') != 'Node' or e.get('ph') != 'X' or not e['name'].endswith('_kernel_time'):
            continue
        index = bisect_right(starts, e['ts'])-1
        assert index >= 0 and e['ts']+e['dur'] <= runs[index]['ts']+runs[index]['dur'], e['name']
        assert e['pid'] == runs[index]['pid'] and e['tid'] == runs[index]['tid']
        assert e['args']['provider'] == 'CPUExecutionProvider'
        groups[index].append(e)
    assert set(groups) == set(range(len(calls))), 'Missing operator coverage'
    aggregates = defaultdict(lambda: dict(calls=0, inclusive_us=0, exclusive_us=0))
    phase_totals = dict(warmup=dict(run_us=0, node_us=0), measured=dict(run_us=0, node_us=0))
    shapes = Counter()
    nodes = {}
    for index, (run, call) in enumerate(zip(runs, calls, strict=True)):
        request = requests[call['request']]
        phase = 'warmup' if request['iteration'] == 0 else 'measured'
        phase_totals[phase]['run_us'] += run['dur']
        for event, own in exclusive(groups[index]):
            args = event['args']; name = event['name'][:-len('_kernel_time')]
            phase_totals[phase]['node_us'] += own
            node_info = dict(op=args['op_name'], provider=args['provider'], node_index=args.get('node_index'))
            assert name not in nodes or nodes[name] == node_info, 'Ambiguous node identity'
            nodes[name] = node_info
            shape = json.dumps(dict(inputs=args.get('input_type_shape'), outputs=args.get('output_type_shape')), sort_keys=True)
            shapes[(name, shape)] += 1
            if phase == 'measured':
                key = (args['op_name'], name)
                row = aggregates[key]
                row['calls'] += 1; row['inclusive_us'] += event['dur']; row['exclusive_us'] += own
    for v in phase_totals.values():
        assert 0 <= v['node_us'] <= v['run_us']
        v['outside_nodes_us'] = v['run_us']-v['node_us']
    return dict(session_calls=len(runs), nodes=nodes, phase_totals=phase_totals,
        categories=dict(Counter(str((e.get('cat'), e.get('ph'))) for e in events)),
        node_clocks=[dict(op=k[0], name=k[1], **v) for k, v in sorted(aggregates.items(), key=lambda p: -p[1]['exclusive_us'])],
        shapes=[dict(name=k[0], **json.loads(k[1]), calls=v) for k, v in shapes.items()])


def phases(observation, result, manifest):
    requests, calls = observation['requests'], observation['calls']
    assert len(requests) == len(result['records']) == 80 and len(calls) == 4960
    totals = defaultdict(lambda: dict(wall_ns=0, cpu_ns=0, calls=0))
    request_totals = defaultdict(int)
    for index, (request, original) in enumerate(zip(requests, result['records'], strict=True)):
        case = manifest['cases'][index % 20]
        assert request['index'] == index and request['name'] == original['name'] == case['name']
        assert request['iteration'] == original['pass'] == index//20
        assert original['start_ticks'] <= request['start_ns'] <= request['end_ns'] <= original['end_ticks']
        group = calls[request['first_call']:request['first_call']+request['calls']]
        assert [g['graph'] for g in group] == ['frontend', 'encoder']+['decoder']*case['expected']['decoder_calls']
        assert request['decoder_calls'] == case['expected']['decoder_calls']
        assert all(c['request'] == index for c in group)
        assert all(a['end_ns'] <= b['start_ns'] for a, b in zip(group, group[1:]))
        assert request['start_ns'] <= group[0]['start_ns'] and group[-1]['end_ns'] <= request['end_ns']
        if request['iteration'] == 0:
            continue
        for call in group:
            row = totals[call['graph']]
            row['wall_ns'] += call['end_ns']-call['start_ns']; row['cpu_ns'] += call['cpu_ns']; row['calls'] += 1
        request_totals['wall_ns'] += original['end_ticks']-original['start_ticks']
        request_totals['observed_cpu_ns'] += request['cpu_ns']
    remainder = request_totals['wall_ns']-sum(v['wall_ns'] for v in totals.values())
    assert remainder >= 0
    return dict(phases=dict(totals), requests=dict(request_totals), remainder_wall_ns=remainder,
                corpus_seconds=request_totals['wall_ns']/3e9,
                corpus_phase_seconds={k: v['wall_ns']/3e9 for k, v in totals.items()},
                corpus_remainder_seconds=remainder/3e9)


def main():
    assert not (BASE/'closed.json').exists()
    folder = BASE/'collected'; receipt = read(folder/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0
    assert read(BASE/'transfer.json')['collection'] == pin(folder/'collection.json')
    assert read(BASE/'transfer.json')['archive'] == pin(BASE/'results.tar.gz')
    for name, wanted in receipt['files'].items():
        assert pin(folder/name) == wanted, name
    spec = read(folder/'spec.json'); state = read(folder/'state.json')
    assert pin(folder/'spec.json') == pin(BASE/'spec.json') == state['spec']
    assert spec['previous_closure'] == pin(APP/'closed.json')
    for name, wanted in spec['tools'].items():
        assert pin(folder/name) == pin(BASE/name) == wanted
    assert state['complete'] and state['code'] == 0 and [r['mode'] for r in state['runs']] == ['control', 'profile']
    assert state['supervisor'] == read(BASE/'deployment.json')
    assert receipt['identities'] == [state['supervisor']]+[r['owner'] for r in state['runs']]
    manifest = read(APP/'collected/manifests/current-parakeet.json')
    module_spec = importlib.util.spec_from_file_location('native_records', APP/'collected/runtime/protocol.py')
    protocol = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(protocol)
    module_spec = importlib.util.spec_from_file_location('foreign_cpu', APP/'collected/runtime/campaign_processes.py')
    accounting = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(accounting)
    results, observations, measured = {}, {}, {}
    for row in state['runs']:
        mode = row['mode']
        assert row['complete'] and row['code'] == 0 and row['seconds'] < 900
        assert row['preflight']['available'] >= 12*1024**3 and row['preflight']['tmpfs'] >= 3*1024**3
        samples = [json.loads(s) for s in (folder/(mode+'.resources.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] and max(s['rss'] for s in samples) == row['peak_rss']
        gaps = [samples[0]['seconds']]+[b['seconds']-a['seconds'] for a, b in zip(samples, samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0 <= g < 10 for g in gaps)
        for sample in samples:
            assert sample['rss'] < 12*1024**3 and sample['available'] >= 1024**3 and sample['tmpfs'] >= 1024**3
            assert sample['output'] <= 1024**3 and sample['seconds'] < 900
            assert all(a == [2] for a in sample['affinities'])
        assert row['accounting']['valid'] and row['accounting']['foreign_cpu_fraction'] <= .01
        assert row['accounting'] == accounting.foreign_fraction(row['before'], row['after'], state['supervisor']['pid'])
        results[mode] = read(folder/mode/'requests/result.json')
        result = results[mode]
        protocol.validate_records(result, manifest, 'timing')
        assert result['manifest_sha256'] == spec['manifest']['sha256']
        assert result['runner_sha256'] == spec['native_consumer']['sha256']
        assert result['adapter_sha256'] == manifest['adapter']['sha256']
        payload = read(APP/'payload.json')
        assert result['python_binary'] == payload['interpreter']
        assert result['native_binaries'] == manifest['native_binaries'] and result['versions'] == manifest['native_versions']
        for path, wanted in result['numeric_libraries'].items():
            assert payload['external'][path] == wanted
        for index, record in enumerate(result['records']):
            assert read(folder/mode/'requests'/f'{index:03}.json') == record
        observation = observations[mode] = read(folder/mode/'observation.json')
        assert observation['mode'] == mode and not observation['graph_outputs_changed']
        measured[mode] = phases(observation, result, manifest)
    assert [r['result'] for r in results['control']['records']] == [r['result'] for r in results['profile']['records']]
    profiled = observations['profile']; profile = {}
    assert set(profiled['profiles']) == {'frontend', 'encoder', 'decoder'}
    assert observations['control']['profiles'] == {}
    for graph, value in profiled['profiles'].items():
        path = folder/'profile'/value['file']
        assert pin(path) == {k: value[k] for k in ['bytes', 'sha256']}
        events = read(path)
        profile[graph] = profile_graph(events, [c for c in profiled['calls'] if c['graph'] == graph], profiled['requests'])
    baseline = next(r for r in read(APP/'analysis.json')['table'] if r['is_corpus'])['ort']['seconds']
    analysis = dict(passed=True, phases=measured, profiles=profile,
        original_native_seconds=baseline, control_over_original=measured['control']['corpus_seconds']/baseline,
        profile_over_control=measured['profile']['corpus_seconds']/measured['control']['corpus_seconds'],
        build_info=profiled['build_info'], capabilities=read(folder/'capabilities.json'),
        benchmark_update=False, attribution_only=True)
    write(BASE/'analysis.json', analysis)
    write(BASE/'closed.json', dict(passed=True, analysis=pin(BASE/'analysis.json'), transfer=pin(BASE/'transfer.json'),
                                  collection=pin(folder/'collection.json'), terminal_owners=receipt['identities']))
    print(json.dumps({k: analysis[k] for k in ['passed', 'phases', 'control_over_original', 'profile_over_control', 'build_info', 'capabilities']}))


if __name__ == '__main__':
    main()
