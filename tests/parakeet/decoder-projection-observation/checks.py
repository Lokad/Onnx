"""Validate original-graph observations independently of any packing hypothesis."""
from protocol import pin, read
from events import calls


def result(base, name, spec, owner, built):
    folder = base/name
    value = read(folder/'result.json')
    assert value['mode'] == ('control' if name == 'control-run' else 'trace')
    counts = calls(value)
    assert value['pid'] == owner['pid'] and value['runtime'] == '10.0.8'
    assert set(value['isa']) == {'vector_float_count', 'avx2', 'fma', 'avx512'}
    assert value['isa']['vector_float_count'] in [4, 8, 16]
    assert all(type(value['isa'][k]) is bool for k in ['avx2', 'fma', 'avx512'])
    assert value['options'] == dict(optimization='Memory', simd=True, intrinsics=value['isa']['fma'],
                                    max_degree=1, buffer_pool_disabled=False)
    assert value['core_sha256'] == spec['product']['Lokad.Onnx.dll']['sha256']
    assert value['consumer_sha256'] == built['consumer']['sha256']
    assert value['spec_sha256'] == pin(base/'observation.json')['sha256']
    fixture = read(base/'observation.json')
    assert fixture['core_sha256'] == value['core_sha256']
    assert value['model_sha256'] == fixture['model_sha256']
    assert set(value['original_outputs']) == set(fixture['outputs']) and len(value['original_outputs']) == 4
    assert value['mapping'] == value['mapping_after']
    mapping = value['mapping']
    assert mapping['shared_with_execution'] and mapping['budget_bytes'] == 64*1024**2
    assert 0 <= mapping['retained_bytes'] <= mapping['budget_bytes']
    assert mapping['source']['dtype'] == 'Float' and mapping['source']['shape'] == [640, 8198]
    assert mapping['source']['strides'] == [8198, 1] and not mapping['source']['reversed']
    assert mapping['source_array_offset'] == 0 and mapping['source_array_count'] == 640*8198
    assert type(mapping['present']) is bool
    if mapping['present']:
        entry = mapping['entry']
        assert entry['source_name'] == 'onnx::MatMul_230' and entry['source_length'] == 640*8198
        assert entry['source_array_key_exact'] and entry['source_reference_exact'] and entry['packed_reference_exact']
        assert entry['packed']['shape'] == [640, 8198] and entry['packed']['dtype'] == 'Float'
    else:
        assert mapping['entry'] is None
    operands = value['operands']
    assert operands['original_weight_reference'] and operands['b'] == mapping['source']
    for key, shape in [('a', [1, 1, 1, 640]), ('b', [640, 8198]), ('output', [1, 1, 1, 8198])]:
        assert operands[key]['dtype'] == 'Float' and operands[key]['shape'] == shape
    assert pin(folder/'projection-a.f32') == dict(bytes=2560, sha256=operands['a']['sha256'])
    if name == 'trace-capture':
        ready, enabled = read(folder/'ready.json'), read(folder/'collector-enabled.json')
        assert ready['pid'] == enabled['pid'] == value['pid']
        assert ready['native_thread'] == value['native_thread']
        assert ready['counter'] < enabled['counter'] < value['clocks'][0]['marker']
        assert (folder/'capture.nettrace').stat().st_size > 0
        control = read(base/'control-run/result.json')
        for key in ['core_sha256', 'consumer_sha256', 'model_sha256', 'spec_sha256',
                    'original_node_sha256', 'original_outputs', 'mapping', 'operands', 'isa', 'options']:
            assert control[key] == value[key], key
    else:
        assert not (folder/'ready.json').exists() and not (folder/'capture.nettrace').exists()
    return dict(**counts, prepared_mapping_present=mapping['present'])
