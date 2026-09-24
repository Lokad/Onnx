"""Join actual packing counters to complete managed and native projection calls."""
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from review_current_gap import ROOT, OUT, pin, read

BASE = ROOT/'artifacts/parakeet-projection-route-resume-amd-20260924'
ORIGINAL = ROOT/'artifacts/parakeet-projection-route-amd-20260924'
NATIVE = ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924'
LIGHT = ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924'


def main():
    closure = read(BASE/'closed.json'); analysis = read(BASE/'analysis.json')
    assert closure['passed'] and analysis['passed'] and analysis['exact_public_results']
    for name in ['analysis','observations','clocks']:
        assert closure[name] == pin(BASE/(name+'.json'))
    assert analysis['observations'] == 17360 and analysis['requests'] == 160
    assert not analysis['actual_kernel_leaf_observed'] and not analysis['overhead_subtracted']
    native_closure = read(NATIVE/'closed.json')
    assert native_closure['passed'] and native_closure['analysis'] == pin(NATIVE/'analysis.json')
    assert pin(NATIVE/'closed.json')['sha256'] == '615353b4d8524f8935076357c0949859553c27826d50d53ebd82fb002d146a85'
    native_collection = read(NATIVE/'collected/collection.json')
    assert native_closure['collection'] == pin(NATIVE/'collected/collection.json')
    native_observation_path = NATIVE/'collected/profile/observation.json'
    assert native_collection['files']['profile/observation.json'] == pin(native_observation_path)
    native_observation = read(native_observation_path)
    profile_path = NATIVE/'collected/profile'/native_observation['profiles']['encoder']['file']
    assert native_collection['files'][profile_path.relative_to(NATIVE/'collected').as_posix()] == pin(profile_path)
    native_events = read(profile_path)
    native_summary = read(NATIVE/'analysis.json')
    native_runs = sorted([e for e in native_events if e.get('name') == 'model_run' and e.get('ph') == 'X'],key=lambda e:e['ts'])
    native_calls = [r for r in native_observation['calls'] if r['graph'] == 'encoder']
    assert len(native_runs) == len(native_calls) == 80
    assert all(a['ts']+a['dur'] <= b['ts'] for a,b in zip(native_runs,native_runs[1:]))
    starts = [r['ts'] for r in native_runs]
    gap = read(OUT/'current-gap-20260924.json')
    assert gap['passed'] and gap['encoder_descriptors_checked'] == 2856
    groups = {r['managed_nodes'][0]:r for r in gap['projections']}
    native_names = {r['ort_name'] for r in groups.values()}
    assert len(groups) == len(native_names) == 217
    native_clocks = {r['name']:r for r in native_summary['profiles']['encoder']['node_clocks']}
    assert all(native_clocks[n]['inclusive_us'] == native_clocks[n]['exclusive_us'] for n in native_names)
    matched = {}
    for event in native_events:
        if event.get('cat') != 'Node' or event.get('ph') != 'X' or not event['name'].endswith('_kernel_time'):
            continue
        name = event['name'][:-len('_kernel_time')]
        if name not in native_names: continue
        index = bisect_right(starts,event['ts'])-1; run = native_runs[index]
        assert index >= 0 and event['ts']+event['dur'] <= run['ts']+run['dur']
        assert event['pid'] == run['pid'] and event['tid'] == run['tid']
        assert event['args']['provider'] == 'CPUExecutionProvider'
        assert event['args']['op_name'] in ['MatMul','FusedMatMul']
        key = (native_calls[index]['request'],name)
        assert key not in matched; matched[key] = event
    assert len(matched) == 17360
    graph = read(ORIGINAL/'bundle/evidence/graphs.json')['encoder-model.onnx']['nodes']
    graph_ids = {n['name']:n['id'] for n in graph}
    light_closure = read(LIGHT/'closed.json')
    assert light_closure['passed'] and light_closure['analysis'] == pin(LIGHT/'analysis.json')
    assert pin(LIGHT/'closed.json')['sha256'] == '8a6f509a210641650a9c057f472417dde7b4acc320f86eeb3d34ecbced4cc2e6'
    light_folder = LIGHT/'capture-collected'
    assert light_closure['collection'] == pin(light_folder/'capture-collection.json')
    light_receipt = read(light_folder/'capture-collection.json')
    for name in ['wall/result.json','wall/graphs.json']:
        assert light_receipt['files'][name] == pin(light_folder/name)
    light_results = read(light_folder/'wall/result.json')
    assert light_results['core_sha256'] == read(ORIGINAL/'build-collected/built.json')['core']['sha256']
    assert read(light_folder/'wall/graphs.json')['encoder-model.onnx']['nodes'] == graph
    observed_results = read(BASE/'capture-collected/phase/result.json')
    records = read(BASE/'observations.json')
    by_request = defaultdict(list)
    for row in records: by_request[row['request']].append(row)
    assert set(by_request) == set(range(80))
    all_members = [n for g in groups.values() for n in g['managed_nodes']]
    assert len(all_members) == len(set(all_members)) == 265
    buckets = defaultdict(list); details = []; mismatches = []
    for index, rows in sorted(by_request.items()):
        native_request = native_observation['requests'][index]
        raw_path = BASE/'capture-collected/phase'/f'projection-{index:03}.json'
        receipt = read(BASE/'capture-collected/capture-collection.json')
        assert receipt['files'][raw_path.relative_to(BASE/'capture-collected').as_posix()] == pin(raw_path)
        raw = read(raw_path); call, = raw['Calls']
        nodes = {n['NodeId']:n for n in call['Nodes']}
        light_name = f'wall/phase-{index:03}.json'
        assert light_receipt['files'][light_name] == pin(light_folder/light_name)
        light = read(light_folder/light_name)
        assert (light['name'],light['pass'],light['frequency']) == (raw['Name'],raw['Pass'],1000000000)
        assert light_results['records'][index]['result'] == observed_results['records'][index]['result']
        assert light_results['records'][index]['input_sha256'] == observed_results['records'][index]['input_sha256']
        light_call, = [c for c in light['calls'] if c['graph']=='encoder-model.onnx']
        light_nodes = {n['NodeId']:n for n in light_call['nodes']}
        assert list(light_nodes) == list(nodes)
        assert len(rows) == 217 and len(nodes) == 2856
        assert native_request['index'] == index and native_request['name'] == raw['Name']
        assert native_request['iteration'] == raw['Pass'] == index//20
        for row in rows:
            group = groups[row['name']]
            native = matched[index,group['ort_name']]
            assert row['clip'] == native_request['name'] and row['pass_index'] == native_request['iteration']
            assert [row['k'],row['n']] == group['shape']
            positional = row['name'].endswith('/self_attn/linear_pos/MatMul')
            assert row['m'] == (2*row['frames']-1 if positional else row['frames'])
            input_shapes = native['args']['input_type_shape']
            assert len(input_shapes) == 1 and 'float' in input_shapes[0]
            assert input_shapes[0]['float'][-2:] == [row['m'],row['k']]
            expected_scratch = 0 if row['mapped'] and row['row_guard_allows'] else row['k']*row['n']*4
            expected_copy = row['m']*row['k']*4 if positional else 0
            if row['scratch_bytes'] != expected_scratch or row['copy_bytes'] != expected_copy:
                mismatches.append(dict(request=index,name=row['name'],expected_scratch=expected_scratch,
                    actual_scratch=row['scratch_bytes'],expected_copy=expected_copy,copy_bytes=row['copy_bytes']))
            route = ('mapped-allowed' if row['row_guard_allows'] else 'mapped-declined') if row['mapped'] else 'unmapped'
            ticks = sum(nodes[graph_ids[n]]['EndTicks']-nodes[graph_ids[n]]['StartTicks'] for n in group['managed_nodes'])
            light_ticks = sum(light_nodes[graph_ids[n]]['EndTicks']-light_nodes[graph_ids[n]]['StartTicks'] for n in group['managed_nodes'])
            own_ticks = nodes[graph_ids[row['name']]]['EndTicks']-nodes[graph_ids[row['name']]]['StartTicks']
            assert own_ticks == row['node_ticks'] and ticks >= own_ticks
            detail = dict(request=index,clip=row['clip'],pass_index=row['pass_index'],name=row['name'],
                native_name=group['ort_name'],m=row['m'],k=row['k'],n=row['n'],alpha=group['alpha'],route=route,positional=positional,frames=row['frames'],
                scratch_bytes=row['scratch_bytes'],copy_bytes=row['copy_bytes'],
                managed_seconds=light_ticks/1e9,observed_seconds=ticks/1e9,ort_seconds=native['dur']/1e6)
            details.append(detail)
            if row['phase'] != 'measured': continue
            assert row['pass_index'] > 0
            for key in [('route',route),('shape-route',row['k'],row['n'],group['alpha'],route),
                        ('rows-route',row['m'],route),('positional-route',positional,route),('node',row['name']),('total',)]:
                buckets[key].append(detail)
    aggregates = []
    for key, values in sorted(buckets.items(),key=lambda item:str(item[0])):
        managed = sum(r['managed_seconds'] for r in values)/3
        native = sum(r['ort_seconds'] for r in values)/3
        aggregates.append(dict(key=list(key),measurements=len(values),nodes=len({r['name'] for r in values}),
            frames=sorted({r['frames'] for r in values}),rows=sorted({r['m'] for r in values}),managed_seconds=managed,earlier_ort_seconds=native,
            observed_seconds=sum(r['observed_seconds'] for r in values)/3,
            diagnostic_difference=managed-native,scratch_bytes_per_corpus=sum(r['scratch_bytes'] for r in values)//3,
            copy_bytes_per_corpus=sum(r['copy_bytes'] for r in values)//3))
    total, = [r for r in aggregates if r['key'] == ['total']]
    assert total['measurements'] == 13020 and total['nodes'] == 217
    assert math.isclose(total['managed_seconds'],sum(r['managed'] for r in groups.values()),abs_tol=1e-10)
    assert math.isclose(total['earlier_ort_seconds'],sum(r['ort'] for r in groups.values()),abs_tol=1e-10)
    for name, group in groups.items():
        observed, = [r for r in aggregates if r['key'] == ['node',name]]
        assert observed['measurements'] == 60 and math.isclose(observed['earlier_ort_seconds'],group['ort'],abs_tol=1e-12)
        assert math.isclose(observed['managed_seconds'],group['managed'],abs_tol=1e-12)
    result = dict(passed=True,diagnostic_only=True,new_candidate_selected=False,actual_kernel_leaf_observed=False,
        no_new_inference=True,ort_profile_is_earlier=True,overhead_subtracted=False,
        captures=analysis,source_closures=dict(managed=pin(BASE/'closed.json'),ort=pin(NATIVE/'closed.json'),lighter_managed=pin(LIGHT/'closed.json')),
        route_and_lighter_times_are_separate_captures=True,
        route_timing_join='Same unchanged Core, node/input/clip/pass identities; route metadata observed in the heavier capture, clocks reused from the qualified lighter profile.',
        initial_join_refusal=pin(BASE/'initial-join-refusal.json'),initial_join_reviewer=pin(BASE/'initial-join-review.py'),
        current_gap=pin(OUT/'current-gap-20260924.json'),native_profile=pin(profile_path),
        graph=pin(ORIGINAL/'bundle/evidence/graphs.json'),reviewer=pin(Path(__file__)),
        source_files={n:pin(ROOT/'src/Lokad.Onnx'/n) for n in ['GraphPacking.cs','TensorOps.MatMul.cs',
            'Zzz.WideProjectionEntry.cs','Zzz.IsolatedShortMatMul.cs','MathOps.PackedAvx512.cs','AblationSwitches.cs']},
        observed_calls=len(details),complete_matched_nodes=265,route_counter_mismatches=mismatches,
        layout_census=[dict(layout=json.loads(k),calls=v) for k,v in Counter(json.dumps(r['layouts'],sort_keys=True) for r in records).items()],
        aggregates=aggregates)
    with (BASE/'matched-projection-calls.json').open('x',encoding='utf8') as stream: json.dump(details,stream,allow_nan=False)
    result['every_matched_call'] = pin(BASE/'matched-projection-calls.json')
    with (OUT/'projection-routes-20260924.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(total=total,route_counter_mismatches=mismatches,
        routes=[r for r in aggregates if r['key'][0]=='route'],captures=analysis['corpus_seconds'],
        overhead=analysis['observation_over_control'],report=pin(OUT/'projection-routes-20260924.json'))))


if __name__ == '__main__': main()
