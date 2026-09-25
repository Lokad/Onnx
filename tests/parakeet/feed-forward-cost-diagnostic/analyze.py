"""Account for every phase, node and stage without subtracting observer overhead."""
from collections import Counter, defaultdict
import json
import math

MODES = ['clock', 'stages', 'markers']
PHASES = {'nemo128.onnx': 'frontend', 'encoder-model.onnx': 'encoder', 'decoder_joint-model.onnx': 'decoder'}
LABELS = {0: 'Math', 1: 'Copy', 2: 'CopyX', 3: 'CopyY', 4: 'Broadcast',
          5: 'ValidateArguments', 6: 'CalculateIndices', 7: 'Cast', 8: 'GraphOrchestration',
          1000: 'ScratchRent', 1001: 'WeightPacking', 1002: 'PackedMultiplication',
          1003: 'ScratchReturn', 1004: 'OddRowRemainder', 1005: 'OutputPreparation', 1006: 'PreparedMultiplication'}
STAGE_FREQUENCY = 10_000_000


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def integer(value, minimum=0):
    assert type(value) is int and value >= minimum
    return value


def node_cost(node, wall, frequency, stage_frequency=STAGE_FREQUENCY):
    integer(frequency, 1)
    assert stage_frequency == STAGE_FREQUENCY
    ticks = Counter()
    counts = Counter()
    assert node['stages'], 'Missing original graph-orchestration stage'
    for row in node['stages']:
        stage = integer(row['stage'])
        assert stage in LABELS, ('Unrecognized stage', stage)
        ticks[stage] += integer(row['duration_ticks'])
        counts[stage] += 1
    assert node['stages'][0]['stage'] == 8
    elapsed = None
    residual = None
    if wall is not None:
        assert wall['NodeId'] == node['id']
        elapsed = integer(wall['EndTicks']) - integer(wall['StartTicks'])
        integer(elapsed)
        # Compare integer products before conversion to seconds. Do not clamp a
        # negative residual, round it away, or silently absorb uncovered time.
        remainder = elapsed * stage_frequency - sum(ticks.values()) * frequency
        assert remainder >= 0, 'Stage sum exceeds its enclosing node interval'
        residual = remainder / (frequency * stage_frequency)
    return dict(stage_ticks=dict(ticks), stage_counts=dict(counts),
                stage_seconds={LABELS[k]: v / stage_frequency for k, v in ticks.items()},
                wall_ticks=elapsed, wall_seconds=None if elapsed is None else elapsed / frequency,
                residual_seconds=residual)


def feed_forward_cost(node, wall, route, group, node_by_name, frequency, mode):
    assert node['op'] == 'MatMul' and route['m'] == route['frames']
    assert route['name'] == group['managed_nodes'][0] and route['native_name'] == group['ort_name']
    assert [route['k'], route['n']] == group['shape'] and route['alpha'] == group['alpha']
    # TensorNameDesc ends with dtype:dimensionxdimension. Names may contain ::.
    operands = [item.rsplit(':', 2) for item in node['detail'].split(',')]
    assert len(operands) == 2 and all(len(p) == 3 and p[1] == 'float' for p in operands)
    dimensions = [[int(d) for d in p[2].split('x')] for p in operands]
    assert dimensions == [[1, route['m'], route['k']], [route['k'], route['n']]], 'Unexpected actual feed-forward shape'
    cost = node_cost(node, wall, frequency)
    counts = Counter(cost['stage_counts'])
    prepared = route['route'] == 'mapped-allowed'
    if mode == 'markers':
        # Memory-mode MatMul allocates/rents cleared output, then reaches the
        # later conditional-clear boundary even though clearDestination=false.
        assert counts[1005] == 2
        assert counts[1006] == int(prepared)
        assert all(counts[k] == int(not prepared) for k in range(1000, 1005))
        assert not any(counts[k] for k in [1, 2, 3]), 'Unexpected copy stage'
    else:
        assert mode == 'stages' and wall is None
        assert not any(k >= 1000 for k in counts)
    scale = group['managed_nodes'][1:]
    assert len(scale) == (1 if route['alpha'] == .5 else 0)
    scale_seconds = 0.
    scale_stage_seconds = 0.
    for name in scale:
        scaled = node_by_name[name]
        assert scaled['node']['op'] == 'Mul'
        scaled_cost = scaled['cost']
        scale_stage_seconds += sum(scaled_cost['stage_ticks'].values()) / STAGE_FREQUENCY
        if mode == 'markers':
            assert scaled_cost['wall_seconds'] is not None
            scale_seconds += scaled_cost['wall_seconds']
    return dict(request=route['request'], clip=route['clip'], pass_index=route['pass_index'],
        name=route['name'], native_name=route['native_name'], m=route['m'], k=route['k'], n=route['n'],
        alpha=route['alpha'], previous_route=route['route'], prior_scratch_bytes=route['scratch_bytes'],
        prior_copy_bytes=route['copy_bytes'], earlier_ort_seconds=route['ort_seconds'],
        stage_seconds=cost['stage_seconds'], stage_counts=cost['stage_counts'],
        matmul_wall_seconds=cost['wall_seconds'], matmul_residual_seconds=cost['residual_seconds'],
        scale_nodes=scale, scale_stage_seconds=scale_stage_seconds,
        scale_wall_seconds=scale_seconds if mode == 'markers' else None,
        complete_group_seconds=(cost['wall_seconds'] + scale_seconds) if mode == 'markers' else None)


def role(result, folder, mode, expected_graphs, routes, groups, expected_numeric_ops):
    assert mode in MODES
    metadata = read(folder / 'graphs.json')
    assert set(metadata) == set(expected_graphs) == set(PHASES)
    for name, graph in metadata.items():
        assert graph['nodes'] == expected_graphs[name]['nodes']
        for key in ['retained_packed_bytes', 'maximum_packed_bytes']:
            assert graph[key] == expected_graphs[name][key]
    graph_nodes = {g: {n['id']: n for n in v['nodes']} for g, v in metadata.items()}
    assert all(len(graph_nodes[g]) == len(v['nodes']) for g, v in metadata.items())
    assert len(graph_nodes['encoder-model.onnx']) == 2856
    by_name = {n['name']: n for n in metadata['encoder-model.onnx']['nodes']}
    assert len(by_name) == 2856
    assert len(groups) == 96
    members = [n for g in groups.values() for n in g['managed_nodes']]
    assert len(members) == len(set(members)) == 144
    assert all(by_name[name]['op'] == 'MatMul' for name in groups)
    lookup = {(r['request'], r['name']): r for r in routes}
    assert len(lookup) == len(routes) == 7680
    records = result['records']
    assert len(records) == 80
    assert {p.name for p in folder.glob('cost-*.json')} == {f'cost-{i:03}.json' for i in range(80)}
    frequency = None
    requests, projections, accounting = [], [], []
    node_totals = defaultdict(lambda: dict(calls=0, stage_ticks=Counter(), wall_ticks=0, residual_seconds=0.))
    corpus_ticks = 0
    phase_totals = Counter()
    route_counts = Counter()
    for index, record in enumerate(records):
        value = read(folder / f'cost-{index:03}.json')
        assert (value['name'], value['pass'], value['frequency'], value['mode']) == (record['name'], record['pass'], record['frequency'], mode)
        assert value['stage_frequency'] == STAGE_FREQUENCY
        assert frequency in [None, record['frequency']]
        frequency = integer(record['frequency'], 1)
        assert record['pass'] == index // 20
        assert (record['phase'] == 'warmup') == (record['pass'] == 0)
        expected = ['nemo128.onnx', 'encoder-model.onnx'] + ['decoder_joint-model.onnx'] * record['result']['decoder_calls']
        assert [c['graph'] for c in value['calls']] == expected
        previous = record['start_ticks']
        phase_ticks = Counter()
        encoder_costs = None
        phase_residuals = {}
        counters = []
        for call in value['calls']:
            graph = call['graph']
            start, end = integer(call['start_ticks']), integer(call['end_ticks'])
            assert previous <= start < end <= record['end_ticks']
            previous = end
            phase_ticks[PHASES[graph]] += end - start
            assert call['retained_packed_bytes'] == metadata[graph]['retained_packed_bytes']
            counters.append(dict(graph=graph, retained=call['retained_packed_bytes'],
                scratch=integer(call['scratch_bytes']), copies=integer(call['copy_bytes'])))
            if mode == 'clock' or graph != 'encoder-model.onnx':
                assert call['nodes'] is None and call['wall'] is None
                continue
            assert [n['id'] for n in call['nodes']] == list(graph_nodes[graph])
            if mode == 'markers':
                assert [n['NodeId'] for n in call['wall']] == list(graph_nodes[graph])
            else:
                assert call['wall'] == []
            walls = {n['NodeId']: n for n in call['wall']}
            last = start
            wall_sum = stage_sum = 0
            encoder_costs = {}
            for node in call['nodes']:
                descriptor = graph_nodes[graph][node['id']]
                assert node['op'] == descriptor['op']
                wall = walls.get(node['id'])
                if wall is not None:
                    assert last <= wall['StartTicks'] <= wall['EndTicks'] <= end
                    assert wall['Op'] == expected_numeric_ops[descriptor['name']]
                    last = wall['EndTicks']
                cost = node_cost(node, wall, frequency)
                if mode == 'stages':
                    assert all(k < 1000 for k in cost['stage_ticks'])
                stage_sum += sum(cost['stage_ticks'].values())
                wall_sum += cost['wall_ticks'] or 0
                encoder_costs[descriptor['name']] = dict(node=node, cost=cost, wall=wall)
                if record['pass'] > 0:
                    total = node_totals[descriptor['name']]
                    total['calls'] += 1
                    total['stage_ticks'].update(cost['stage_ticks'])
                    total['wall_ticks'] += cost['wall_ticks'] or 0
                    total['residual_seconds'] += cost['residual_seconds'] or 0.
            assert (end - start) * STAGE_FREQUENCY >= stage_sum * frequency
            assert end - start >= wall_sum
            phase_residuals[graph] = dict(stage_seconds=stage_sum / STAGE_FREQUENCY,
                graph_minus_stages_seconds=((end-start)*STAGE_FREQUENCY-stage_sum*frequency)/(frequency*STAGE_FREQUENCY),
                node_seconds=wall_sum/frequency if mode == 'markers' else None,
                graph_minus_nodes_seconds=(end-start-wall_sum)/frequency if mode == 'markers' else None)
        if mode != 'clock':
            assert encoder_costs is not None
            for name, group in groups.items():
                prior = lookup[index, name]
                assert (prior['clip'], prior['pass_index'], prior['frames']) == (record['name'], record['pass'], record['result']['encoded_frames'])
                current = encoder_costs[name]
                projection = feed_forward_cost(current['node'], current['wall'], prior, group, encoder_costs, frequency, mode)
                projections.append(projection)
                if mode == 'markers':
                    route_counts[prior['route']] += 1
        duration = integer(record['end_ticks']) - integer(record['start_ticks'])
        assert duration > 0 and math.isclose(record['seconds'], duration / frequency, rel_tol=1e-13)
        remainder = duration - sum(phase_ticks.values())
        assert remainder >= 0
        if record['pass'] > 0:
            corpus_ticks += duration
            phase_totals.update(phase_ticks)
        requests.append(dict(index=index, name=record['name'], pass_index=record['pass'], seconds=duration/frequency,
            phase_seconds={k: v/frequency for k, v in phase_ticks.items()}, outside_graph_seconds=remainder/frequency,
            encoder_reconciliation=phase_residuals))
        accounting.append(counters)
    if mode == 'markers':
        assert route_counts == Counter({'mapped-allowed': 468, 'mapped-declined': 252, 'unmapped': 6960})
    assert len(projections) == (0 if mode == 'clock' else 7680)
    assert len(node_totals) == (0 if mode == 'clock' else 2856)
    node_rows = []
    for name, total in node_totals.items():
        assert total['calls'] == 60
        node_rows.append(dict(name=name, op=by_name[name]['op'], calls=60,
            stage_seconds={LABELS[k]: ticks/(3*STAGE_FREQUENCY) for k, ticks in total['stage_ticks'].items()},
            wall_seconds=total['wall_ticks']/(3*frequency) if mode == 'markers' else None,
            residual_seconds=total['residual_seconds']/3 if mode == 'markers' else None))
    return dict(corpus_seconds=corpus_ticks/(3*frequency), frequency=frequency,
        phase_seconds={k: v/(3*frequency) for k, v in phase_totals.items()}, requests=requests,
        graph_accounting=accounting, node_rows=node_rows, projections=projections,
        exact_route_counts=dict(route_counts), no_overhead_subtracted=True)


def controls(roles):
    assert set(roles) == set(MODES)
    rows = []
    for mode, analysis in roles.items():
        measured = [r for r in analysis['requests'] if r['pass_index'] > 0]
        assert len(measured) == 60
        assert len({(r['name'], r['pass_index']) for r in measured}) == 60
        passes = [sum(r['seconds'] for r in measured if r['pass_index'] == p) for p in [1, 2, 3]]
        assert all(math.isfinite(x) and x > 0 for x in passes)
        ratio = max(passes) / min(passes)
        rows.append(dict(name=mode + ':corpus-repeatability', values=passes, ratio=ratio, limit=1.10, passed=ratio <= 1.10))
        names = sorted({r['name'] for r in measured})
        assert len(names) == 20
        for name in names:
            values = [r['seconds'] for r in measured if r['name'] == name]
            assert len(values) == 3 and all(math.isfinite(x) and x > 0 for x in values)
            assert {r['pass_index'] for r in measured if r['name'] == name} == {1, 2, 3}
            ratio = max(values) / min(values)
            rows.append(dict(name=mode + ':' + name + ':repeatability', values=values, ratio=ratio, limit=1.20, passed=ratio <= 1.20))
    effects = []
    for numerator, denominator in [('stages', 'clock'), ('markers', 'stages')]:
        ratio = roles[numerator]['corpus_seconds'] / roles[denominator]['corpus_seconds']
        assert math.isfinite(ratio) and ratio > 0
        effects.append(dict(name=numerator + '/' + denominator, ratio=ratio,
            minimum=.95, maximum=1.05, passed=.95 <= ratio <= 1.05))
    assert len(rows) == 63 and len(effects) == 2
    return dict(repeatability=rows, observer_effects=effects,
                usable_for_candidate_selection=all(r['passed'] for r in rows + effects))


def families(markers):
    grouped = defaultdict(list)
    for row in markers['projections']:
        if row['pass_index'] > 0:
            grouped[row['k'], row['n'], row['alpha'], row['previous_route']].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        stages = Counter()
        for row in rows:
            stages.update(row['stage_seconds'])
        result.append(dict(k=key[0], n=key[1], alpha=key[2], route=key[3], calls_per_corpus=len(rows)//3,
            complete_seconds=sum(r['complete_group_seconds'] for r in rows)/3,
            earlier_ort_seconds=sum(r['earlier_ort_seconds'] for r in rows)/3,
            stages={k: v/3 for k, v in stages.items()},
            transition_residual_seconds=sum(r['matmul_residual_seconds'] for r in rows)/3,
            scale_seconds=sum(r['scale_wall_seconds'] for r in rows)/3,
            prior_scratch_bytes_per_corpus=sum(r['prior_scratch_bytes'] for r in rows)//3))
    assert sum(r['calls_per_corpus'] for r in result) == 1920
    for row in result:
        accounted = sum(row['stages'].values()) + row['transition_residual_seconds'] + row['scale_seconds']
        assert math.isclose(accounted, row['complete_seconds'], rel_tol=1e-12, abs_tol=1e-12)
    return result
