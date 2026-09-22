"""Complete inherited raw/layer assertions, extended only for the wide family."""
from protocol import read


def raw_check(result, lanes, core, wide):
    assert result['core'] == core and result['lanes'] == lanes
    assert {r['m'] for r in result['observations']} == ({64, 128} if wide else {32, 48})
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
            test = i-2648; assert (row['c'], row['m'], row['h'], row['w'], row['stride']) == (32, 64 if wide else 32, 3, 7, 1)
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


def layers_check(result, lanes, original, fixture):
    assert result['passed'] and result['cases'] == len(result['observations']) == 108
    assert result['eligible'] == 96 and result['fallback'] == 12
    assert result['failed'] == result['differences'] == result['native_failures'] == 0
    assert result['values'] == 119823360 and result['observations'] == original['observations']
    assert result['maximum'] == original['maximum'] <= 1e-4
    assert result['read_only_operands'] and result['layer_graphs'] == 108 and result['graph_retained_bytes'] == 3*21086208
    assert result['prepared_weights'] == 32 and result['prepared_bytes'] == original['prepared_bytes']
    assert result['lanes'] == lanes and len(result['graph_dispatch']) == 216
    for i, row in enumerate(result['graph_dispatch']):
        call = fixture['calls'][i//2]
        assert (row['name'], row['index'], row['form'], row['eligible'], row['repeat']) == (call['case'], call['index'], call['form'], call['eligible'], i % 2)
        x, y = call['input']['shape'], call['output']['shape']
        if row['eligible']:
            assert row['scratch_bytes'] == (x[1]*(x[2]+2)*(x[3]+2)+y[1]*y[2]*y[3])*4
            assert row['prepared_bytes'] == call['weights']['bytes']
        else: assert row['prepared_bytes'] == 0 and row['scratch_bytes'] > 0
        residual = call['residual'] is not None
        expected = ['ConvRelu' if call['relu'] and not residual else 'Conv']
        if residual: expected.append('AddRelu' if call['relu'] else 'Add')
        assert row['nodes'] == expected
    return dict(passed=True, cases=108, values=119823360, graph_calls=216, native_maximum=result['maximum'], lanes=lanes)


def check_result(result, mode, width, spec, base):
    assert result['passed'] and result['core'] == spec['core']['sha256']
    assert result['executable'] == spec['consumers'][mode]['sha256']
    assert result['flags'] == (['DOTNET_EnableAVX512'] if width == '256' else [])
    assert result['avx512'] == (width == '512') and result['no_performance_measurement']
    original = read(base/('windows-'+mode+'.json'))
    assert result['observations'] == original['observations']
    lanes = int(width)//32
    if mode in ['raw', 'wide']:
        analysis = raw_check(result, lanes, spec['core']['sha256'], mode == 'wide')
        for actual, expected in zip(result['supplemental'], original['supplemental'], strict=True):
            assert {k:v for k,v in actual.items() if k != 'nan_payload_differences'} == {k:v for k,v in expected.items() if k != 'nan_payload_differences'}
        return analysis
    assert mode == 'layers'
    return layers_check(result, lanes, original, read(base/'fixtures/result.json'))
