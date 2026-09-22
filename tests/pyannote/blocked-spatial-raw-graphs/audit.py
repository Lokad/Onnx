"""Audit all raw graph cases, arithmetic observations, resources and identities."""
import json
from build import BASE, CORE, COMPONENT, TOOLS, pin, read, save, verify, terminal, prior
from transform import transform


def check(result, lanes):
    assert result['core'] == CORE and result['lanes'] == lanes
    assert (result['geometries'], result['layout_cases'], result['cases']) == (312, 331, 2648)
    assert result['finite_kernel_cases'] == 2496 and result['nonfinite_fallback_cases'] == 152
    assert result['rejected'] == 10 and result['owned_outputs'] and result['no_performance_measurement']
    assert len(result['observations']) == 2648 and len(result['supplemental']) == 20
    assert result['values'] == sum(r['values'] for r in result['observations'])
    assert result['scalar_differences'] == sum(r['scalar_differences'] for r in result['observations'])
    assert result['production_differences'] == sum(r['production_differences'] for r in result['observations'])
    assert result['failed_cases'] == sum(bool(r['scalar_differences'] or r['production_differences']) for r in result['observations']) + sum(r['differences'] != 0 for r in result['supplemental'])
    cases = result['graph_cases']; assert len(cases) == result['graph_controls'] == 2668 and result['graph_candidates'] == 5336
    assert [(r['family'], r['index']) for r in cases] == [('raw', i) for i in range(2648)]+[('supplemental', i) for i in range(20)]
    total = payloads = finite = 0
    for i, row in enumerate(cases):
        if i < 2648:
            old = result['observations'][i]
            assert all(row[k] == old[k] for k in ['c', 'm', 'h', 'w', 'stride', 'relu', 'values'])
            assert row['bias'] == old['hasBias'] and row['residual'] == old['hasResidual']
            assert row['finite_convolution'] == (not old['special'])
        else:
            test = i-2648; assert (row['c'], row['m'], row['h'], row['w'], row['stride']) == (32, 32, 3, 7, 1)
            assert row['bias'] and row['residual'] and row['relu'] == bool(test & 1)
            assert row['finite_convolution'] == (test >= 12)
        c, m, h, w, stride = (row[k] for k in ['c', 'm', 'h', 'w', 'stride'])
        oh, ow = (h+stride-1)//stride, (w+stride-1)//stride
        assert row['values'] == m*oh*ow
        assert row['retained_weights'] == ((m+2*lanes-1)//(2*lanes)*2*lanes)*c*9*4
        layout = (c*(h+2)*(w+2)+m*oh*ow)*4
        assert row['control_scratch'] > 0
        assert row['expected_scratch'] == layout+(0 if row['finite_convolution'] else row['control_scratch'])
        assert [r['repeat'] for r in row['executions']] == [0, 1]
        for execution in row['executions']:
            assert execution['scratch_bytes'] == row['expected_scratch']
            assert execution['differences'] >= 0 and execution['nan_payload_differences'] >= 0
            total += execution['differences']; payloads += execution['nan_payload_differences']
        expected = ['ConvRelu' if row['relu'] and not row['residual'] else 'Conv']
        if row['residual']: expected.append('AddRelu' if row['relu'] else 'Add')
        assert row['nodes'] == expected; finite += row['finite_convolution']
    assert finite == 2504 and result['graph_differences'] == total and result['graph_nan_payload_differences'] == payloads
    assert result['passed'] == (result['failed_cases'] == 0 and total == 0)
    return dict(passed=result['passed'], raw_cases=2648, supplemental=20, rejected=10,
        graph_cases=2668, control_requests=2668, candidate_requests=5336, finite_graphs=2504, fallback_graphs=164,
        differences=total, nan_payload_differences=payloads, raw_nan_payload_differences=sum(r['nan_payload_differences'] for r in result['supplemental']))


def main():
    assert not (BASE/'closed.json').exists(); prior()
    value = read(BASE/'verified.json'); verify(value['files'])
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] in [0, 1]
    assert [r['name'] for r in state['runs']] == ['restore', 'build', 'raw-256']
    identities = [state['supervisor']]; resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] in ([0, 1] if row['name'] == 'raw-256' else [0]) and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if row['name'] == 'raw-256' else 8)*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for s in samples:
            assert s['seconds'] < 900 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
            assert s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3 and s['rss'] == sum(m['rss'] for m in s['members'])
            for m in s['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: terminal(identity)
    assert pin(BASE/'output/256.json') == value['report']
    result = read(BASE/'output/256.json'); analysis = check(result, 8)
    assert result['passed'] == value['passed'] == (state['code'] == 0)
    assert result['pid'] == state['runs'][-1]['worker']['pid'] and result['executable'] == value['consumer']['sha256']
    assert result['runtime'] == '10.0.12' and not result['flags'] and not result['avx512']
    output, diff = transform((COMPONENT/'source/Probe.cs').read_text(), CORE)
    assert (BASE/'source/Probe.cs').read_text() == output and (BASE/'consumer.diff').read_text() == diff
    assert pin(BASE/'source/GraphRaw.cs') == pin(TOOLS/'GraphRaw.cs.txt')
    analysis.update(resources=resources, core=value['core'], consumer=value['consumer'], no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=result['passed'], files=files, local_inputs=value['files'], identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()
