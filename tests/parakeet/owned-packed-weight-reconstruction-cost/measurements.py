"""Check every original contract and account for the retained stage/encoder clocks."""
from fractions import Fraction

JOBS = ['selected-512-a', 'candidate-512-a', 'candidate-512-b', 'selected-512-b']
ROLES = ['selected', 'candidate']
METRICS = ['encoder', 'feed_forward', 'copy_y', 'math', 'scales']


def integer(value, minimum=0):
    assert type(value) is int and value >= minimum
    return value


def qualify(result, spec, role, consumer):
    assert role in ROLES and result['role'] == role and result['mode'] == '512'
    assert result['passed'] and result['diagnostic_only'] and result['runtime'] == '.NET 10.0.8'
    assert result['processor_count'] == 1 and result['affinity'] == 4 and result['avx512']
    assert result['immutable_inputs'] and result['held_outputs_independent']
    assert not result['product_rebuilt'] and not result['application_scored'] and not result['forced_gc']
    assert result['passes'] == 4 and result['warmup_passes'] == 1
    assert result['core_sha256'] == spec['products'][role]['Lokad.Onnx.dll']['sha256']
    assert result['data_sha256'] == spec['products'][role]['Lokad.Onnx.Data.dll']['sha256']
    assert result['runner_sha256'] == consumer['sha256']
    evidence = spec['diagnostic_evidence']
    assert result['node_manifest'] == evidence['graph_manifest']
    nodes = result['node_manifest']
    by_name = {n['name']: n for n in nodes}
    weights = {w['Name']: w for w in spec['weights']}
    groups = evidence['groups']
    members = [name for group in groups.values() for name in group['managed_nodes']]
    assert len(nodes) == len(by_name) == 2856 and len(groups) == len(weights) == 96
    assert len(members) == len(set(members)) == 144
    projection_ids = {by_name[name]['id'] for name in groups}
    scale_ids = {by_name[name]['id'] for group in groups.values() for name in group['managed_nodes'][1:]}
    assert len(projection_ids) == 96 and len(scale_ids) == 48 and not projection_ids & scale_ids
    references = evidence['reference_outputs'][role]
    assert len(references) == len(spec['cases']) == 20 and len(result['records']) == 80
    clocks = []
    previous_end = 0
    for index, record in enumerate(result['records']):
        pass_index, case_index = divmod(index, 20)
        case = spec['cases'][case_index]
        original = references[case_index]
        assert record['passed'] and record['request_index'] == index and record['pass'] == pass_index
        assert record['phase'] == ('warmup' if pass_index == 0 else 'measured')
        assert all(record[key] == value for key, value in original.items())
        assert record['name'] == case['name'] and record['input_hash'] == case['raw_sha256']
        frames = case['expected']['encoded_frames']
        assert record['frames'] == frames and record['remainder'] == (frames % 2 != 0 and frames % 3 != 0)
        start, end = integer(record['encoder_start'], 1), integer(record['encoder_end'], 1)
        frequency = integer(record['encoder_frequency'], 1)
        assert previous_end < start < end
        previous_end = end
        assert record['profile_frequency'] == 10000000
        assert [n['node_id'] for n in record['node_times']] == [n['id'] for n in nodes]
        times = {n['node_id']: integer(n['ticks']) for n in record['node_times']}
        assert len(times) == 2856
        total = sum(times.values())
        assert total * frequency <= (end - start) * 10000000, 'Profiler stages exceed enclosing encoder clock'
        calls = record['calls']
        assert len(calls) == 96 and {c['node'] for c in calls} == set(groups)
        assert {c['node_id'] for c in calls} == projection_ids
        assert {c['weight'] for c in calls} == set(weights)
        copy_ticks = math_ticks = copy_count = 0
        for call in calls:
            group = groups[call['node']]
            node = by_name[call['node']]
            weight = weights[call['weight']]
            assert node['id'] == call['node_id'] and node['op'] == 'MatMul' and node['inputs'][1] == call['weight']
            assert call['b'] == weight['Shape'] == group['shape']
            assert call['a'] in [[frames, call['b'][0]], [1, frames, call['b'][0]]]
            owned = role == 'candidate' and not weight['Cached']
            assert call['owned'] == owned
            expected_copy = int(owned and record['remainder'])
            assert call['copy_y_stages'] == expected_copy
            copied = integer(call['copy_y_ticks'])
            assert (copied > 0) == bool(expected_copy)
            math = integer(call['math_ticks'], 1)
            node_ticks = integer(call['node_ticks'], 1)
            assert node_ticks == times[call['node_id']] and copied + math <= node_ticks
            copy_ticks += copied
            math_ticks += math
            copy_count += expected_copy
        assert copy_count == (87 if role == 'candidate' and record['remainder'] else 0)
        scales = sum(times[n] for n in scale_ids)
        ff = sum(times[n] for n in projection_ids) + scales
        assert 0 <= copy_ticks < ff <= total and 0 < math_ticks + scales <= ff
        clocks.append(dict(index=index, name=record['name'], frames=frames, pass_index=pass_index,
                           phase=record['phase'], encoder_ticks=end-start, encoder_frequency=frequency,
                           profile_ticks=total, feed_forward_ticks=ff, copy_y_ticks=copy_ticks,
                           math_ticks=math_ticks, scale_ticks=scales, profile_frequency=10000000,
                           reconstructions=copy_count, scratch_bytes=record['scratch_bytes'], copy_bytes=record['copy_bytes']))
    assert sum(row['reconstructions'] for row in clocks) == (2436 if role == 'candidate' else 0)
    return clocks


def seconds(row, metric):
    if metric == 'encoder':
        return Fraction(row['encoder_ticks'], row['encoder_frequency'])
    key = 'scale_ticks' if metric == 'scales' else metric + '_ticks'
    return Fraction(row[key], row['profile_frequency'])


def repeatability(first, second, corpus):
    assert first > 0 and second > 0
    ratio = max(first, second) / min(first, second)
    limit = Fraction(110 if corpus else 120, 100)
    return dict(ratio=float(ratio), limit=float(limit), passed=ratio <= limit)


def summarize(clocks, cases):
    assert list(clocks) == JOBS and len(cases) == 20
    names = [c['name'] for c in cases]
    assert len(set(names)) == 20
    for rows in clocks.values():
        assert len(rows) == 80 and [r['name'] for r in rows] == names * 4
        assert [r['phase'] for r in rows] == ['warmup'] * 20 + ['measured'] * 60
    controls = []
    table = []
    for name in [*names, 'complete-corpus']:
        corpus = name == 'complete-corpus'
        scope = set(names) if corpus else {name}
        row = dict(name=name, corpus=corpus)
        for role in ROLES:
            processes = [key for key in JOBS if key.startswith(role + '-')]
            values = {}
            for metric in METRICS:
                means = [sum(seconds(r, metric) for r in clocks[job] if r['phase'] == 'measured' and r['name'] in scope) / 3 for job in processes]
                average = sum(means) / 2
                values[metric] = dict(seconds=float(average), numerator=average.numerator, denominator=average.denominator,
                                      processes=[dict(job=job, seconds=float(value), numerator=value.numerator, denominator=value.denominator)
                                                 for job, value in zip(processes, means, strict=True)])
                if metric in ['encoder', 'feed_forward']:
                    controls.append(dict(name=name, role=role, metric=metric, **repeatability(*means, corpus)))
                if metric == 'copy_y':
                    expected_nonzero = role == 'candidate' and (corpus or next(c['expected']['encoded_frames'] for c in cases if c['name'] == name) % 2 != 0
                                                               and next(c['expected']['encoded_frames'] for c in cases if c['name'] == name) % 3 != 0)
                    if expected_nonzero:
                        controls.append(dict(name=name, role=role, metric=metric, **repeatability(*means, corpus)))
                    else:
                        assert means == [0, 0], 'Unexpected reconstruction clock'
            row[role] = values
        table.append(row)
    assert len(controls) == 92
    for role in ROLES:
        for metric in METRICS:
            total = sum(Fraction(r[role][metric]['numerator'], r[role][metric]['denominator']) for r in table[:-1])
            assert total == Fraction(table[-1][role][metric]['numerator'], table[-1][role][metric]['denominator'])
    return dict(usable_for_attribution=all(r['passed'] for r in controls), controls=controls, table=table,
                application_scored=False, release_admitted=False, overhead_subtracted=False,
                policy='Four fresh normal-mode processes; one warmup and three measured complete encoder passes. Corpus max/min <=1.10 and each clip <=1.20. No trimming or unchanged retry.')
