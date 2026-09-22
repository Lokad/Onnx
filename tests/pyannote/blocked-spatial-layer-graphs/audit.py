"""Reconcile every ordinary layer-graph output and dispatch observation."""
import json
from build import BASE, COMPONENT, FIXTURES, PRODUCT, TOOLS, pin, read, save, verify, terminal
from transform import transform


def main():
    assert not (BASE/'closed.json').exists()
    value = read(BASE/'verified.json'); verify(value['files'])
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] in [0, 1]
    assert [r['name'] for r in state['runs']] == ['restore', 'build', 'graphs-256']
    resources = []; identities = [state['supervisor']]
    for row in state['runs']:
        assert row['complete'] and row['code'] in ([0, 1] if row['name'] == 'graphs-256' else [0]) and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if row['name'] == 'graphs-256' else 8)*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for s in samples:
            assert s['seconds'] < 900 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
            assert s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3 and s['rss'] == sum(m['rss'] for m in s['members'])
            for m in s['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
            if row['name'] == 'graphs-256': assert len(s['members']) <= 1
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: terminal(identity)
    assert pin(BASE/'output/256.json') == value['report']
    result = read(BASE/'output/256.json'); assert result['passed'] == value['passed'] == (state['code'] == 0)
    assert result['pid'] == state['runs'][-1]['worker']['pid'] and result['core'] == value['core']['sha256'] and result['executable'] == value['consumer']['sha256']
    assert not result['flags'] and result['lanes'] == 8 and not result['avx512'] and result['runtime'] == '10.0.12'
    assert result['cases'] == len(result['observations']) == 108 and result['values'] == 119823360
    assert result['differences'] == sum(r['differences'] for r in result['observations'])
    assert result['native_failures'] == sum(r['native_failed'] for r in result['observations'])
    assert result['failed'] == sum(r['differences'] != 0 or r['native_failed'] != 0 for r in result['observations'])
    assert result['passed'] == (result['failed'] == 0)
    if result['passed']: assert result['observations'] == read(COMPONENT/'output/model-256.json')['observations']
    assert result['read_only_operands'] and result['layer_graphs'] == 108 and result['graph_retained_bytes'] == 3*21086208
    assert result['eligible'] == 96 and result['fallback'] == 12 and len(result['graph_dispatch']) == 216
    calls = read(FIXTURES/'output/result.json')['calls']
    for i, row in enumerate(result['graph_dispatch']):
        call = calls[i//2]
        assert (row['name'], row['index'], row['form'], row['eligible'], row['repeat']) == (call['case'], call['index'], call['form'], call['eligible'], i % 2)
        x, y = call['input']['shape'], call['output']['shape']
        if call['eligible']:
            assert row['scratch_bytes'] == (x[1]*(x[2]+2)*(x[3]+2)+y[1]*y[2]*y[3])*4
            assert row['prepared_bytes'] == call['weights']['bytes']
        else: assert row['prepared_bytes'] == 0 and row['scratch_bytes'] > 0
        has_residual = call['residual'] is not None
        expected = ['ConvRelu' if call['relu'] and not has_residual else 'Conv']
        if has_residual: expected.append('AddRelu' if call['relu'] else 'Add')
        assert row['nodes'] == expected
    for name in ['BlockedSpatial.cs', 'GeneratedKernels.cs', 'VectorInput.cs', 'VectorEpilogue.cs']:
        assert pin(BASE/'source'/name) == pin(COMPONENT/'source'/name)
    source, diff = transform((COMPONENT/'source/ModelProbe.cs').read_text(), value['core']['sha256'])
    assert (BASE/'source/ModelProbe.cs').read_text() == source and (BASE/'consumer.diff').read_text() == diff
    assert pin(BASE/'source/GraphCalls.cs') == pin(TOOLS/'GraphCalls.cs.txt')
    analysis = dict(passed=result['passed'], cases=108, values=result['values'], exact_selected_values=result['differences'] == 0,
        native_failures=result['native_failures'], native_maximum=result['maximum'], graph_calls=216,
        eligible_graphs=96, fallback_graphs=12, resources=resources, no_performance_measurement=True,
        whole_models_qualified=False, core=value['core'], consumer=value['consumer'])
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=result['passed'], layer_graphs_only=True, files=files, local_inputs=value['files'], identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()
