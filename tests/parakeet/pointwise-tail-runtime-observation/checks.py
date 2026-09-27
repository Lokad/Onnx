"""Preserve every original shape, output and timer/allocation check."""
from protocol import pin, read


def result(base, spec, row, built):
    folder = base/'trace-capture'; value = read(folder/'output/result.json')
    original = read(base/'spec.json')
    assert value['completed'] and value['passed'] and value['diagnostic_only']
    assert value['pid'] == row['processes']['worker']['pid'] and value['role'] == 'candidate'
    assert value['runtime'] == '.NET 10.0.8' and value['flags'] == []
    assert value['core_sha256'] == spec['product']['Lokad.Onnx.dll']['sha256']
    assert value['consumer_sha256'] == built['consumer']['sha256']
    assert value['spec_sha256'] == pin(base/'spec.json')['sha256']
    assert value['inputs_immutable'] and value['frequency'] > 0
    assert original['warmups'] == original['measurements'] == 5 and len(original['shapes']) == 40
    expected = [(phase, repeat, index, shape) for phase in ['warmup', 'measured']
        for repeat in range(5) for index, shape in enumerate(original['shapes'])]
    assert len(value['records']) == len(value['diagnostics']) == len(expected) == 400
    previous = value['started_ticks']; total = 0
    for ordinal, (clock, diagnostic, (phase, repeat, index, shape)) in enumerate(zip(value['records'], value['diagnostics'], expected, strict=True)):
        assert (clock['phase'], clock['pass'], clock['index']) == (phase, repeat, index)
        assert {k:clock[k] for k in shape} == shape
        assert clock['output_sha256'] == spec['output_hashes'][index] and clock['bit_exact']
        assert clock['allocated_bytes'] == 0 and clock['ticks'] > 0
        assert (diagnostic['ordinal'], diagnostic['phase'], diagnostic['repeat'], diagnostic['call']) == (ordinal, phase, repeat, index)
        assert previous < diagnostic['marker'] <= diagnostic['start'] < diagnostic['stop'] <= diagnostic['afterMarker']
        assert clock['ticks'] == diagnostic['stop']-diagnostic['start']
        for generation in range(3): assert 0 <= diagnostic['gc'+str(generation)] <= diagnostic['after'+str(generation)]
        assert 0 <= diagnostic['pauseBefore'] <= diagnostic['pauseAfter']
        previous = diagnostic['afterMarker']; total += clock['ticks']
    assert value['ended_ticks'] > previous and value['ended_ticks']-value['started_ticks'] >= total
    ready, enabled = read(folder/'ready.json'), read(folder/'collector-enabled.json')
    assert ready['pid'] == enabled['pid'] == value['pid']
    assert ready['native_thread'] == value['native_thread']
    assert ready['counter'] < enabled['counter'] < value['diagnostics'][0]['marker']
    assert (folder/'capture.nettrace').stat().st_size > 0
    return dict(passed=True, diagnostic_only=True, calls=400, output_arrays=400,
        warmup=200, measured=200, actual_product=spec['product'], result=pin(folder/'output/result.json'))
