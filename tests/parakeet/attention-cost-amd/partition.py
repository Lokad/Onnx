"""Require exact current attention coverage and disjoint per-call stage costs."""
from collections import Counter, defaultdict


def partition(costs, records, manifest, expected):
    assert costs['passed'] and costs['protocol'] == 'parakeet-attention-cost-v1'
    assert costs['frequency'] == 1_000_000_000 and costs['runtime'] == '10.0.8'
    assert costs['processor_count'] == 1 and costs['fma'] and costs['avx2'] and costs['avx512'] and not costs['flags']
    stages = ['rent', 'pack', 'arithmetic', 'return', 'final_row']
    assert costs['stages'] == stages
    rows = costs['rows']
    assert len(rows) == len(expected['rows']) == 9600 and len(records) == 80
    frames = {c['name']: c['expected']['encoded_frames'] for c in manifest['cases']}
    wanted = {}
    for r in expected['rows']:
        key = r['request'], r['weight']
        assert key not in wanted
        wanted[key] = r
    selected, groups = [], defaultdict(list)
    for index, request in enumerate(records):
        calls = rows[index*120:(index+1)*120]
        assert {(index,r['weight']) for r in calls} == {k for k in wanted if k[0] == index}
        for offset, row in enumerate(calls):
            meta = wanted[index, row['weight']]
            assert meta['clip'] == request['name'] and meta['phase'] == request['phase']
            assert meta['pass_index'] == index//20
            m = 2*frames[request['name']]-1 if '/linear_pos/' in meta['node'] else frames[request['name']]
            assert row['index'] == index*120+offset and row['m'] == m == meta['m']
            assert row['mapped'] == (meta['route'] != 'unmapped')
            assert row['row_guard_allows'] == (m%2 == 0 or m%3 == 0)
            fallback = meta['route'] != 'mapped-allowed'
            assert row['fallback'] == fallback
            assert request['start_ticks'] <= row['start_ticks'] <= row['end_ticks'] <= request['end_ticks']
            if row['index']: assert rows[row['index']-1]['end_ticks'] <= row['start_ticks']
            main_rows = (m if m%3 == 0 else m-m%2) if fallback else 0
            assert row['main_rows'] == main_rows
            counts = [1,1,1,1,int(main_rows != m)] if fallback else [0]*5
            assert row['stage_calls'] == counts
            assert all(len(row[k]) == 5 for k in ['stage_ticks','stage_starts','stage_ends'])
            previous = row['start_ticks']
            for i, count in enumerate(counts):
                start, end, ticks = row['stage_starts'][i], row['stage_ends'][i], row['stage_ticks'][i]
                if count:
                    assert previous <= start <= end <= row['end_ticks'] and ticks == end-start
                    previous = end
                else: assert start == end == ticks == 0
            assert sum(row['stage_ticks']) <= row['end_ticks']-row['start_ticks']
            assert row['simd'] and row['intrinsics'] and row['degree'] == 1
            if fallback:
                assert row['leaf'] == ('ShortWideMultiply3Rows' if m%3 == 0 else 'ShortWideMultiply2Rows')
                assert row['scratch_elements'] >= 1024**2
            else:
                assert row['leaf'] == '' and row['scratch_elements'] == 0
            if request['phase'] == 'measured':
                selected.append(row)
                groups[('route',meta['route'])].append(row)
                groups[('route-rows',meta['route'],m)].append(row)
                groups[('total',)].append(row)
    assert len({r['thread'] for r in rows}) == 1 and len(selected) == 7200
    assert {k[1]:len(v) for k,v in groups.items() if k[0]=='route'} == {
        'mapped-allowed':957,'mapped-declined':723,'unmapped':5520}
    result = []
    for key, group in sorted(groups.items(),key=lambda x:str(x[0])):
        ticks = [sum(r['stage_ticks'][i] for r in group) for i in range(5)]
        total = sum(r['end_ticks']-r['start_ticks'] for r in group)
        result.append(dict(group=list(key),measured_calls=len(group),corpus_seconds=total/3e9,
            stages={k:v/3e9 for k,v in zip(stages,ticks)},remainder_seconds=(total-sum(ticks))/3e9,
            leaves=dict(Counter(r['leaf'] or 'prepared-leaf-not-observed' for r in group)),
            input_layouts=dict(Counter(r['input_layout'] for r in group)),
            weight_layouts=dict(Counter(r['weight_layout'] for r in group))))
    return result
