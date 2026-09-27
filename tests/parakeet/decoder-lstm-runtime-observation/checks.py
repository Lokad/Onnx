"""Require every original call/output and each added diagnostic interval."""
from protocol import pin, read


def result(base, spec, row, built):
    folder = base/'trace-capture'; value = read(folder/'output/result.json')
    owner = row['processes']['worker']
    assert value['passed'] and value['diagnostic_only'] and value['pid'] == owner['pid']
    assert value['role'] == 'selectedfallback' and value['mode'] == '512' and value['worker'] == 'trace-capture'
    assert value['affinity'] == 4 and value['processor_count'] == 1 and value['runtime'] == '10.0.8'
    assert value['avx512'] and value['flags'] == {}
    for field, name in [('core', 'Lokad.Onnx.dll'), ('data', 'Lokad.Onnx.Data.dll')]:
        assert value[field] == spec['product'][name]['sha256']
    assert value['consumer'] == built['consumer']['sha256']
    assert value['spec_sha256'] == pin(base/'spec.json')['sha256']
    assert value['capture_sha256'] == pin(base/'fixtures/result.json')['sha256']
    assert all(value[k] for k in ['inputs_unchanged', 'held_outputs_unchanged', 'exact_selected_outputs',
        'complete_calls', 'setup_includes_graph_preparation_and_context',
        'per_call_includes_reset_validation_allocation_and_execution'])
    assert len(value['setup']) == 2
    for index, setup in enumerate(value['setup']):
        assert setup['index'] == index and setup['ticks'] > 0 and setup['allocated_bytes'] >= 0
        assert setup['retained_bytes'] == 0
    calls = read(base/'fixtures/result.json')['calls']; assert len(calls) == 380
    expected = [(phase, repeat, index, call) for phase in ['warmup', 'measured']
        for repeat in range(5) for index, call in enumerate(calls)]
    assert len(value['rows']) == len(value['diagnostics']) == len(expected) == 3800
    previous = 0
    for ordinal, (clock, diagnostic, (phase, repeat, index, call)) in enumerate(zip(value['rows'], value['diagnostics'], expected, strict=True)):
        assert (clock['phase'], clock['repeat'], clock['name'], clock['step'], clock['index']) == (phase, repeat, call['name'], call['step'], call['index'])
        assert clock['output_sha256'] == [o['sha256'] for o in call['outputs']]
        assert clock['allocated_bytes'] >= 0 and clock['ticks'] > 0
        assert (diagnostic['ordinal'], diagnostic['phase'], diagnostic['repeat'], diagnostic['call']) == (ordinal, phase, repeat, index)
        assert previous < diagnostic['marker'] <= diagnostic['start'] < diagnostic['stop'] <= diagnostic['afterMarker']
        assert clock['ticks'] == diagnostic['stop']-diagnostic['start']
        for generation in range(3): assert 0 <= diagnostic['gc'+str(generation)] <= diagnostic['after'+str(generation)]
        assert 0 <= diagnostic['pauseBefore'] <= diagnostic['pauseAfter']
        previous = diagnostic['afterMarker']
    ready, enabled = read(folder/'ready.json'), read(folder/'collector-enabled.json')
    assert ready['pid'] == enabled['pid'] == value['pid']
    assert ready['native_thread'] == value['native_thread']
    assert ready['counter'] < enabled['counter'] < value['diagnostics'][0]['marker']
    assert (folder/'capture.nettrace').stat().st_size > 0
    return dict(passed=True, diagnostic_only=True, calls=3800, output_arrays=11400,
        warmup=1900, measured=1900, actual_product=spec['product'], result=pin(folder/'output/result.json'))
