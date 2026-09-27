"""Validate observer scope, complete requests and an exhaustive cost partition."""
import base64
import collections
import json
from pathlib import Path
import sys
from run import BASE, ROOT, TOOLS, PRELUDE, pin, read, write, ssh, prepared, load


def collected(kind):
    prepared()
    folder = BASE/(kind+'-collected')
    spec = read(BASE/'bundle/spec.json')
    receipt = read(folder/(kind+'-collection.json'))
    transfer = read(BASE/(kind+'-transfer.json'))
    assert transfer['passed'] and transfer['archive'] == pin(BASE/(kind+'-results.tar.gz'))
    assert transfer['collection'] == pin(folder/(kind+'-collection.json'))
    assert receipt['terminal'] and receipt['code'] == 0
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    assert pin(folder/'spec.json') == pin(BASE/'bundle/spec.json')
    state = read(folder/(kind+'-state.json'))
    assert state['complete'] and state['code'] == 0 and receipt['state'] == pin(folder/(kind+'-state.json'))
    assert state['supervisor'] == read(BASE/(kind+'-deployment.json'))
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p),birth=b)
        for r in state['runs'] for p,b in r['members'].items()]
    names = ['sdk-version','core-restore','core-build','bridge-restore','bridge-build','inventory'] if kind == 'build' else ['control','observed']
    assert [r['name'] for r in state['runs']] == names
    limits = spec[kind+'_limits']
    resources = []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < limits['seconds']
        assert run['preflight']['available'] >= limits['available_before'] and run['preflight']['tmpfs'] >= limits['tmpfs_before']
        samples = [json.loads(s) for s in (folder/'logs'/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0
        for sample in samples:
            assert sample['seconds'] < limits['seconds'] and sample['rss'] < limits['rss']
            assert min(sample['available'],sample['tmpfs']) >= spec['minimum_free'] and sample['output'] < spec['output_limit']
            assert sample['rss'] == sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert run['members'][str(member['pid'])] == member['birth'] and member['affinity'] == [2]
                assert member['threads'] and all(t == [2] for t in member['threads'])
        gaps = [samples[0]['seconds']] + [b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])] + [run['seconds']-samples[-1]['seconds']]
        assert all(0 <= g < 10 for g in gaps)
        resources.append(dict(name=run['name'],samples=len(samples),seconds=run['seconds'],peak_rss=max(s['rss'] for s in samples)))
    built = read(folder/'built.json')
    assert built['passed'] and built['inventory'] == pin(folder/'logs/instructions.json')
    assert built['runtime'].keys() == spec['runtime'].keys()
    for name, wanted in built['runtime'].items(): assert pin(folder/'runtime'/name) == wanted, name
    assert [n for n in built['runtime'] if built['runtime'][n] != spec['runtime'][n]] == ['Lokad.Onnx.dll']
    assert built['product'] == {n:built['runtime'][n] for n in spec['before_product']}
    assert built['consumer'] == spec['consumer']
    return folder, spec, receipt, state, built, resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder, spec, receipt, state, built, resources = collected('build')
    inventory = read(folder/'logs/instructions.json')
    assert inventory['inventory_complete']
    assert [r['assembly'] for r in inventory['observations']] == ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']
    scope = []
    for row in inventory['observations']:
        core = row['assembly'] == 'Lokad.Onnx.dll'
        assert row['before_sha256'] == spec['before_product'][row['assembly']]['sha256']
        assert row['after_sha256'] == built['product'][row['assembly']]['sha256']
        assert row['methods'] == (3286 if core else 697) and not row['removed']
        if core:
            assert len(row['differences']) == 3
            assert {n.split('::')[1] for n in row['differences']} == set(spec['changed_methods'])
            assert all(n.startswith('Lokad.Onnx.Tensor`1[T]::') for n in row['differences'])
            assert row['added'] and all(n.startswith(('Lokad.Onnx.PointwiseCostProbe::','Lokad.Onnx.PointwiseCostProbe+')) for n in row['added'])
        else: assert not row['differences'] and not row['added']
        assert row['unchanged_methods'] == row['methods']-(3 if core else 0)
        assert row['public_surface_equal'] and row['public_surface'] == row['public_surface_after']
        assert row['assembly_attributes_before'] == row['assembly_attributes_after']
        assert all(row['method_flags_after'][n] == v for n,v in row['method_flags_before'].items())
        assert set(row['candidate_methods']) == set(row['differences']+row['added'])
        scope.append(dict(assembly=row['assembly'],unchanged=row['unchanged_methods'],changed=row['differences'],added=row['added']))
    warnings = []
    for run in state['runs']:
        text = (folder/'logs'/(run['name']+'.stdout')).read_text() + (folder/'logs'/(run['name']+'.stderr')).read_text()
        assert ': error ' not in text
        warnings.extend(line for line in text.splitlines() if ': warning ' in line)
    assert len(warnings) == 4 and all('Zzz.WideProjectionEntry.cs(20,' in s and 'warning CS8604:' in s for s in warnings), warnings
    result = dict(passed=True,built=pin(folder/'built.json'),scope=scope,warnings=warnings,resources=resources,
        consumer_unchanged=True,data_unchanged=True,source_reversal_checked=True,diagnostic_only=True,release_admitted=False,
        source=pin(BASE/'bundle/spec.json'),reviewer=pin(Path(__file__)),collection=pin(folder/'build-collection.json'))
    write(BASE/'build-review.json',result)
    encoded = base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    transferred = ssh(PRELUDE+f'''
from remote import verify,live,read,pin
import base64
verify();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r}
with (base/'build-review.json').open('xb') as f:f.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review'] == pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',transferred)
    print(json.dumps(dict(**transferred,scope=scope)))


def partition(costs, records, manifest):
    assert costs['passed'] and costs['protocol'] == 'parakeet-pointwise-cost-v1'
    assert costs['frequency'] == 1000000000 and costs['runtime'] == '10.0.8'
    assert costs['processor_count'] == 1 and costs['fma'] and costs['avx2'] and costs['avx512'] and not costs['flags']
    stages = ['output_initialize','materialize','clear','pack','arithmetic']
    assert costs['stages'] == stages
    rows = costs['rows']
    assert len(rows) == 3840
    frames = {c['name']:c['expected']['encoded_frames'] for c in manifest['cases']}
    selected = []
    for index, request in enumerate(records):
        calls = rows[index*48:(index+1)*48]
        assert [r['filters'] for r in calls] == [2048,1024]*24
        for offset, row in enumerate(calls):
            assert row['index'] == index*48+offset and row['columns'] == frames[request['name']]
            assert request['start_ticks'] <= row['start_ticks'] <= row['end_ticks'] <= request['end_ticks']
            if row['index']: assert rows[row['index']-1]['end_ticks'] <= row['start_ticks']
            assert row['stage_calls'] == [1]*5 and all(t >= 0 for t in row['stage_ticks'])
            assert sum(row['stage_ticks']) <= row['end_ticks']-row['start_ticks']
            assert row['simd'] and row['intrinsics'] and row['degree'] == 1
            assert row['leaf'] == 'mm_unsafe_vectorized_intrinsics_2x4packed_bump'
            assert row['scratch_elements'] >= 1024*row['columns']
            if request['phase'] == 'measured': selected.append(row)
    assert len({r['thread'] for r in rows}) == 1 and len(selected) == 2880
    result = []
    for filters in [2048,1024]:
        group = [r for r in selected if r['filters'] == filters]
        ticks = [sum(r['stage_ticks'][i] for r in group) for i in range(5)]
        total = sum(r['end_ticks']-r['start_ticks'] for r in group)
        result.append(dict(filters=filters,measured_calls=len(group),corpus_seconds=total/3e9,
            stages={k:v/3e9 for k,v in zip(stages,ticks)},remainder_seconds=(total-sum(ticks))/3e9,
            arithmetic_fraction=ticks[-1]/total,
            input_same=sum(r['input_same'] for r in group),weight_same=sum(r['weight_same'] for r in group),
            input_layouts=dict(collections.Counter(r['input_layout'] for r in group)),
            weight_layouts=dict(collections.Counter(r['weight_layout'] for r in group))))
    return result


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources = collected('capture')
    assert pin(folder/'build-review.json') == pin(BASE/'build-review.json')
    assert read(BASE/'build-collected/build-state.json')['ended'] < state['started']
    protocol = load('pointwise_protocol',folder/'protocol.py')
    accounting = load('pointwise_accounting',folder/'campaign_processes.py')
    manifest = read(folder/'manifest.json')
    reference = read(folder/'reference-public.json')
    assert reference['core_sha256'] == spec['before_product']['Lokad.Onnx.dll']['sha256']
    assert reference['data_sha256'] == spec['before_product']['Lokad.Onnx.Data.dll']['sha256']
    expected = {r['name']:r['result'] for r in reference['records']}
    assert len(expected) == 20
    results, times = {}, {}
    for run in state['runs']:
        role = run['name']
        actual = accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
        assert actual == run['accounting'] and actual['valid'] and actual['foreign_cpu_fraction'] <= .01
        result = read(folder/'probe'/role/'result.json')
        protocol.validate_records(result,manifest,'timing')
        assert result['passed'] and not result['sampled'] and not result['flags']
        assert result['runtime'] == '.NET 10.0.8' and result['processor_count'] == 1 and result['held_outputs_unchanged']
        core = spec['before_product']['Lokad.Onnx.dll'] if role == 'control' else built['product']['Lokad.Onnx.dll']
        assert result['core_sha256'] == core['sha256'] and result['data_sha256'] == spec['before_product']['Lokad.Onnx.Data.dll']['sha256']
        assert result['runner_sha256'] == spec['consumer']['sha256'] and result['manifest_sha256'] == pin(folder/'manifest.json')['sha256']
        assert len(result['records']) == 80
        for i,row in enumerate(result['records']):
            assert row == read(folder/'probe'/role/f'{i:03}.json') and row['thread_id'] == run['ready']['thread_id']
            assert row['result'] == expected[row['name']] and row['ownership']
            assert row['input_sha256'] == reference['records'][i]['input_sha256']
        results[role] = result
        times[role] = sum(r['seconds'] for r in result['records'] if r['phase']=='measured')/3
    costs = read(folder/'logs/costs.json')
    assert costs['pid'] == state['runs'][1]['owner']['pid']
    assert costs['core_sha256'] == built['product']['Lokad.Onnx.dll']['sha256']
    groups = partition(costs,results['observed']['records'],manifest)
    ratio = times['observed']/times['control']
    usable = ratio <= spec['observer_over_control_limit']
    analysis = dict(passed=True,usable_for_candidate_selection=usable,corpus_seconds=times,
        observer_over_control=ratio,observer_over_control_limit=spec['observer_over_control_limit'],
        groups=groups,resources=resources,requests=160,observed_pointwise_calls=3840,
        original_public_results_exact=True,diagnostic_only=True,release_admitted=False,
        raw_costs=pin(folder/'logs/costs.json'),reviewer=pin(Path(__file__)))
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,usable_for_candidate_selection=usable,
        analysis=pin(BASE/'analysis.json'),terminal_owners=receipt['identities'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),usable=usable,corpus=times,observer_over_control=ratio,
        groups=[{k:v for k,v in r.items() if 'layouts' not in k} for r in groups])))


if __name__ == '__main__': {'build':build,'capture':capture}[sys.argv[1]]()
