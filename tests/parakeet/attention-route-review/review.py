"""Join revalidated route metadata to current attention clocks, without inference."""
from bisect import bisect_right
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from binding import ROOT, TOOLS, OLD, ROUTES, MANAGED, NATIVE, GAP, bind, pin, read

OUT = ROOT / 'artifacts/parakeet-attention-route-review-20260927'
RESULTS = TOOLS.parent / 'pointwise-tail-profile-results'


def calculate(bound, old_records=None):
    inputs = dict(bound['inputs'])

    def checked(path, wanted):
        assert pin(path) == wanted, path
        inputs[path.relative_to(ROOT).as_posix()] = wanted
        return read(path)

    gap_closed = read(GAP / 'closed.json')
    assert pin(GAP / 'closed.json')['sha256'] == 'b7e24651bfcdcc67900783bb6f33fa215927f20e48f572e3272025fb0b6fbec0'
    assert gap_closed['passed']
    gap = checked(GAP / 'analysis.json', gap_closed['analysis'])
    assert gap['root'] == bound['context']['root'] and gap['application'] == bound['context']['application']
    assert gap['managed_profile'] == pin(MANAGED / 'closed.json')
    assert gap['native_profile'] == pin(NATIVE / 'closed.json')
    assert gap['attribution_only'] and not gap['historical_clocks_used']
    native_closed = read(NATIVE / 'closed.json')
    assert native_closed['passed']
    native_analysis = checked(NATIVE / 'analysis.json', native_closed['analysis'])
    native_receipt = checked(NATIVE / 'collected/collection.json', native_closed['collection'])
    managed_receipt = checked(MANAGED / 'capture-collected/capture-collection.json', read(MANAGED / 'closed.json')['collection'])
    old_receipt = checked(ROUTES / 'capture-collected/capture-collection.json', read(ROUTES / 'closed.json')['collection'])
    old_results = checked(ROUTES / 'capture-collected/phase/result.json', old_receipt['files']['phase/result.json'])
    managed_results = checked(MANAGED / 'capture-collected/wall/result.json', managed_receipt['files']['wall/result.json'])
    observation = checked(NATIVE / 'collected/profile/observation.json', native_receipt['files']['profile/observation.json'])
    profile_name = 'profile/' + observation['profiles']['encoder']['file']
    events = checked(NATIVE / 'collected' / profile_name, native_receipt['files'][profile_name])

    projection = read(RESULTS / 'projection-breakdown-20260927.json')
    assert projection['passed'] and projection['partition'] == pin(GAP / 'closed.json')
    inputs[(RESULTS / 'projection-breakdown-20260927.json').relative_to(ROOT).as_posix()] = pin(RESULTS / 'projection-breakdown-20260927.json')
    selected = {r['native']: r for r in projection['records'] if r['group'].startswith('linear_')}
    assert len(selected) == 120 and all(r['weight']['dims'] == [1024, 1024] for r in selected.values())
    assert all(r['managed_members'] == [name] for name, r in selected.items())
    graph_ids = {n['name']: n['id'] for n in bound['graph']['nodes']}
    source_records = bound['records'] if old_records is None else old_records
    metadata = {}
    cached = {m['Source'] for m in bound['maps']}
    for row in source_records:
        if row['name'] not in selected:
            continue
        key = (row['request'], row['name'])
        assert key not in metadata, 'Duplicate retained route'
        assert row['mapped'] == (selected[row['name']]['weight']['name'] in cached)
        assert row['row_guard_allows'] == (row['m'] % 2 == 0 or row['m'] % 3 == 0)
        assert row['k'] == row['n'] == 1024 and row['m'] >= 51
        assert row['scratch_bytes'] == (0 if row['mapped'] and row['row_guard_allows'] else 4 * 1024**2)
        assert row['layouts'][1]['exact_dense'] and row['layouts'][1]['row_major']
        assert row['layouts'][1]['array_offset'] == 0 and row['layouts'][1]['storage_matches_length']
        # Explicit whitelist: historical node/copy/interval clocks do not enter the join.
        metadata[key] = {k: row[k] for k in ['request', 'clip', 'pass_index', 'phase', 'frames', 'name', 'm', 'mapped', 'row_guard_allows']}
    assert set(metadata) == {(i, name) for i in range(80) for name in selected}, 'Incomplete attention metadata'
    assert len({selected[row['name']]['weight']['name'] for row in metadata.values() if row['mapped']}) == 28

    runs = sorted([e for e in events if e.get('name') == 'model_run' and e.get('ph') == 'X'], key=lambda e: e['ts'])
    calls = [r for r in observation['calls'] if r['graph'] == 'encoder']
    assert len(runs) == len(calls) == 80
    assert all(a['ts'] + a['dur'] <= b['ts'] for a, b in zip(runs, runs[1:]))
    starts = [r['ts'] for r in runs]
    matched = {}
    for event in events:
        if event.get('cat') != 'Node' or event.get('ph') != 'X' or not event.get('name', '').endswith('_kernel_time'):
            continue
        name = event['name'][:-len('_kernel_time')]
        if name not in selected:
            continue
        index = bisect_right(starts, event['ts']) - 1
        assert index >= 0
        run = runs[index]
        assert event['ts'] + event['dur'] <= run['ts'] + run['dur']
        assert event['pid'] == run['pid'] and event['tid'] == run['tid']
        assert event['args']['provider'] == 'CPUExecutionProvider' and event['args']['op_name'] == 'MatMul'
        key = (calls[index]['request'], name)
        assert key not in matched
        matched[key] = event
    assert set(matched) == set(metadata)

    rows = []
    for index in range(80):
        name = f'wall/phase-{index:03}.json'
        phase = checked(MANAGED / 'capture-collected' / name, managed_receipt['files'][name])
        native_request = observation['requests'][index]
        assert native_request['index'] == index and phase['frequency'] == 1_000_000_000
        assert phase['name'] == native_request['name'] and phase['pass'] == native_request['iteration'] == index // 20
        assert old_results['records'][index]['input_sha256'] == managed_results['records'][index]['input_sha256']
        call, = [c for c in phase['calls'] if c['graph'] == 'encoder-model.onnx']
        nodes = {n['NodeId']: n for n in call['nodes']}
        assert len(nodes) == len(call['nodes']) == 2856
        for name in sorted(selected):
            meta, event = metadata[index, name], matched[index, name]
            assert meta['clip'] == phase['name'] and meta['pass_index'] == phase['pass']
            assert meta['phase'] == ('warmup' if index < 20 else 'measured')
            positional = '/linear_pos/' in name
            assert meta['m'] == (2 * meta['frames'] - 1 if positional else meta['frames'])
            assert event['args']['input_type_shape'] == [{'float': [1, meta['m'], 1024]}]
            assert event['args']['output_type_shape'] == [{'float': [1, meta['m'], 1024]}]
            node = nodes[graph_ids[name]]
            assert call['start_ticks'] <= node['StartTicks'] <= node['EndTicks'] <= call['end_ticks']
            route = ('mapped-allowed' if meta['row_guard_allows'] else 'mapped-declined') if meta['mapped'] else 'unmapped'
            rows.append(dict(request=index, clip=meta['clip'], pass_index=meta['pass_index'], phase=meta['phase'],
                node=name, family=selected[name]['group'], rows=meta['m'], frames=meta['frames'], route=route,
                managed_seconds=(node['EndTicks'] - node['StartTicks']) / phase['frequency'], ort_seconds=event['dur'] / 1e6))
    assert len(rows) == 9600 and len({r['frames'] for r in rows}) == 19
    groups = defaultdict(list)
    for row in rows:
        if row['phase'] == 'measured':
            for key in [('route', row['route']), ('family-route', row['family'], row['route']), ('node', row['node']), ('rows-route', row['rows'], row['route']), ('total',)]:
                groups[key].append(row)
    aggregate = []
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        a = sum(r['managed_seconds'] for r in items) / 3
        b = sum(r['ort_seconds'] for r in items) / 3
        aggregate.append(dict(group=list(key), measurements=len(items), nodes=len({r['node'] for r in items}),
                              managed_seconds=a, ort_seconds=b, difference=a-b))
    native_clocks = {r['name']: r for r in native_analysis['profiles']['encoder']['node_clocks']}
    for row in aggregate:
        if row['group'][0] == 'node':
            name = row['group'][1]
            assert row['measurements'] == 60
            assert native_clocks[name]['inclusive_us'] == native_clocks[name]['exclusive_us']
            assert math.isclose(row['managed_seconds'], selected[name]['managed_seconds'], abs_tol=1e-11)
            assert math.isclose(row['ort_seconds'], selected[name]['ort_seconds'], abs_tol=1e-11)
    total, = [r for r in aggregate if r['group'] == ['total']]
    assert total['measurements'] == 7200 and total['nodes'] == 120
    return dict(passed=True, inputs=inputs, root=bound['context']['root'], partition=pin(GAP / 'closed.json'),
        interpretation=bound['interpretation'], source_transformations=bound['transformations'],
        observations=9600, measured_calls=7200, frames=sorted({r['frames'] for r in rows}),
        cached_attention_weights=28, uncached_attention_weights=92, complete_fresh_attention_accounting=True,
        historical_clocks_used=False, historical_copy_costs_used=False, new_inference_calls=0,
        current_internal_route_observed=False, stage_costs_measured=False, new_candidate_selected=False,
        overhead_subtracted=False, aggregate=aggregate), rows


def main(publish=False):
    value, rows = calculate(bind())
    if publish:
        assert not OUT.exists()
        outputs = [RESULTS / 'attention-routes-20260927.json', RESULTS / 'attention-clocks-20260927.csv']
        assert not any(p.exists() for p in outputs)
        OUT.mkdir()
        value['reviewer'] = {p.name: pin(p) for p in TOOLS.glob('*.py')}
        with (OUT / 'analysis.json').open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        with outputs[1].open('x', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        closed = dict(passed=True, analysis=pin(OUT / 'analysis.json'), clocks=pin(outputs[1]), new_inference_calls=0)
        with (OUT / 'closed.json').open('x', encoding='utf8') as stream:
            json.dump(closed, stream, indent=2)
        with outputs[0].open('x', encoding='utf8') as stream:
            json.dump(dict(closure=pin(OUT / 'closed.json'), **value), stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=True, published=publish,
        routes=[r for r in value['aggregate'] if r['group'][0] in ['route', 'total']]), indent=2))


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    main(bool(sys.argv[1:]))
