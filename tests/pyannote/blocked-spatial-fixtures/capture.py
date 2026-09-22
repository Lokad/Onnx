"""Expose only declared layer operands/endpoints; this changes graph optimization."""
import collections
import math
from pathlib import Path
import sys
from run import BASE, ROOT, pin, read, save, verify, monitor


def main():
    spec = read(BASE/'inputs.json'); verify(spec['files'])
    import numpy as np
    import onnx
    import onnxruntime as ort
    assert ort.__version__ == '1.29.0'
    for name in ['numpy', 'onnx', 'onnxruntime']:
        assert str(Path(sys.modules[name].__file__).resolve()) == spec['packages'][name]
    own = monitor.psutil.Process(); assert own.cpu_affinity() == [2]
    output = BASE/'output'; model_path = ROOT/'models/pyannote-embedding/embedding_encoder.onnx'
    model = onnx.load(model_path)
    original_nodes = [n.SerializeToString() for n in model.graph.node]
    original_initializers = [n.SerializeToString() for n in model.graph.initializer]
    main_output, = [n.name for n in model.graph.output]
    for dimension, value in zip(model.graph.input[0].type.tensor_type.shape.dim, [1, 998, 80]): dimension.dim_value = value
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    assert original_nodes == [n.SerializeToString() for n in model.graph.node]
    assert original_initializers == [n.SerializeToString() for n in model.graph.initializer]
    values = {v.name: v for v in [*model.graph.value_info, *model.graph.input, *model.graph.output]}
    shapes = {name: [d.dim_value for d in value.type.tensor_type.shape.dim] for name, value in values.items()}
    weights = {v.name: v for v in model.graph.initializer}
    convs = {n.input[2]: n for n in model.graph.node if n.op_type == 'Conv'}; assert len(convs) == 36
    consumers = collections.defaultdict(list)
    for node in model.graph.node:
        for name in node.input: consumers[name].append(node)
    native = ROOT/'artifacts/pyannote-native-layout-amd-20260922'
    optimized = read(native/'embedding-graph-and-execution.json')['optimized']['nodes']
    nodes = {n['name']: n for n in optimized if n['op'] == 'Conv' and n['domain'] == 'com.microsoft.nchwc'}
    census = read(ROOT/'artifacts/pyannote-blocked-spatial-census-20260922/census.json')
    forms = {f['index']: f for f in census['forms']}; mappings = []; requested = {main_output}
    for call in census['node_calls']:
        node = nodes[call['name']]; conv = convs[node['inputs'][2]]; form = forms[call['form']]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in conv.attribute}
        for key in ['group', 'dilations', 'strides', 'kernel_shape', 'pads']:
            assert attrs[key] == form['attributes'][key], (conv.name, key)
        assert shapes[conv.input[0]] == form['input_shape']
        assert list(weights[conv.input[1]].dims) == form['weight_shape']
        endpoint = conv.output[0]; residual = None
        if form['residual']:
            add, = consumers[endpoint]; assert add.op_type == 'Add' and add.input[0] == endpoint
            residual = add.input[1]; endpoint = add.output[0]
            assert shapes[residual] == form['output_shape']
        relu = form['attributes'].get('activation') == 'Relu'
        assert form['attributes'].get('activation') in [None, 'Relu']
        if relu:
            activation, = consumers[endpoint]; assert activation.op_type == 'Relu'; endpoint = activation.output[0]
        assert shapes[endpoint] == form['output_shape']
        requested.update([conv.input[0], endpoint])
        if residual: requested.add(residual)
        mappings.append(dict(index=len(mappings), node=conv.name, native_node=call['name'], form=call['form'],
            eligible=call['eligible'], input=conv.input[0], weights=conv.input[1], bias=conv.input[2],
            residual=residual, output=endpoint, relu=relu, attributes=attrs))
    assert len(mappings) == 36 and sum(m['eligible'] for m in mappings) == 32
    assert sum(m['residual'] is not None for m in mappings) == 16 and sum(m['relu'] for m in mappings) == 33
    assert len({m['node'] for m in mappings}) == 36
    for name in sorted(requested - {main_output}):
        assert shapes[name] and all(d > 0 for d in shapes[name]); model.graph.output.append(values[name])
    onnx.checker.check_model(model)
    derived = output/'layer-capture.onnx'; onnx.save(model, derived)
    constant_names = sorted({m[k] for m in mappings for k in ['weights', 'bias']})
    constants_bytes = sum(math.prod(weights[n].dims)*4 for n in constant_names)
    expected_bytes = constants_bytes + 3*sum(math.prod(shapes[n])*4 for n in requested) + 3*math.prod(shapes[main_output])*4
    assert expected_bytes + derived.stat().st_size < 1024**3
    declaration = dict(mappings=mappings, shapes={n: shapes[n] for n in sorted(requested)},
        expected_tensor_bytes=expected_bytes, constants_bytes=constants_bytes, derived=pin(derived),
        original=pin(model_path), original_nodes_and_initializers_unchanged=True,
        scope='Fixture capture with extra outputs; graph optimizations may differ. No performance measurement.')
    save(output/'capture-spec.json', declaration)
    arrays = {}
    def store(key, array):
        assert array.dtype == np.float32 and np.isfinite(array).all()
        assert key not in arrays
        target = output/(str(len(arrays)).zfill(4)+'.f32')
        np.asarray(array, dtype='<f4').tofile(target)
        arrays[key] = dict(path=target.name, shape=list(array.shape), **pin(target))
        return arrays[key]
    for name in constant_names: store(name, onnx.numpy_helper.to_array(weights[name]))
    def session(path):
        options = ort.SessionOptions(); options.log_severity_level = 4
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ['session.intra_op.allow_spinning', 'session.inter_op.allow_spinning']: options.add_session_config_entry(key, '0')
        result = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        assert result.get_providers() == ['CPUExecutionProvider']; return result
    def compare(actual, expected):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        scaled = np.abs(actual.astype(np.float64)-expected.astype(np.float64))/np.maximum(1., np.abs(expected.astype(np.float64)))
        return dict(values=int(actual.size), failed_values=int(np.count_nonzero(scaled > 1e-4)),
            maximum=float(scaled.max(initial=0)), identical=actual.tobytes() == expected.tobytes())
    control = session(model_path); capture = session(derived)
    output_names = [v.name for v in capture.get_outputs()]; assert set(output_names) == requested
    cases = [c for c in read(native/'payload/payload.json')['cases'] if c['model'] == 'embedding']
    checks = []; calls = []
    for case in cases:
        x = np.fromfile(native/'payload'/case['input'], dtype='<f4').reshape(case['shape']); before = x.tobytes()
        actual_control, = control.run(None, {control.get_inputs()[0].name: x}); held_control = actual_control.tobytes()
        previous = np.load(native/'payload'/case['reference'], allow_pickle=False)
        check_previous = compare(actual_control, previous)
        store(case['name']+'/unmodified-native', actual_control)
        first = capture.run(None, {capture.get_inputs()[0].name: x})
        first_by_name = dict(zip(output_names, first))
        check_capture = compare(first_by_name[main_output], actual_control)
        for name, value in first_by_name.items():
            assert list(value.shape) == shapes[name]; store(case['name']+'/'+name, value)
        second = capture.run(None, {capture.get_inputs()[0].name: x})
        assert all(a.tobytes() == b.tobytes() == (output/arrays[case['name']+'/'+name]['path']).read_bytes()
                   for name, a, b in zip(output_names, first, second))
        assert x.tobytes() == before and actual_control.tobytes() == held_control
        checks.append(dict(case=case['name'], unmodified_vs_retained=check_previous, captured_vs_unmodified=check_capture,
                           repeated_and_held_exact=True, input_unchanged=True))
        for mapping in mappings:
            row = {k: mapping[k] for k in ['index', 'node', 'native_node', 'form', 'eligible', 'relu', 'attributes']}
            row['case'] = case['name']
            for key in ['input', 'output', 'residual', 'weights', 'bias']:
                name = mapping[key]
                row[key] = None if name is None else arrays[(case['name']+'/' if key in ['input', 'output', 'residual'] else '')+name]
            calls.append(row)
        del first, first_by_name, second, previous, actual_control
    tensor_bytes = sum(r['bytes'] for r in arrays.values()); assert tensor_bytes == expected_bytes
    passed = all(r[k]['failed_values'] == 0 for r in checks for k in ['unmodified_vs_retained', 'captured_vs_unmodified'])
    verify(spec['files']); assert pin(derived) == declaration['derived']
    save(output/'result.json', dict(passed=passed, arrays=len(arrays), tensor_bytes=tensor_bytes, tensors=arrays,
        calls=calls, checks=checks, pid=own.pid, birth=own.create_time(), affinity=own.cpu_affinity(),
        versions=dict(onnx=onnx.__version__, onnxruntime=ort.__version__, numpy=np.__version__),
        settings=dict(provider='CPUExecutionProvider', threads=1, sequential=True, optimization='all', spinning=False),
        capture_spec=pin(output/'capture-spec.json'), no_performance_measurement=True))
    assert passed, checks


if __name__ == '__main__': main()
