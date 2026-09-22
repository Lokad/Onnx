"""Native references for all captured LSTM outputs using the same managed operands."""
import copy
from pathlib import Path
import sys
from run import BASE, ROOT, MODEL, pin, read, save, verify, monitor


def main():
    spec = read(BASE/'inputs.json'); verify(spec['files']); verify(read(BASE/'binaries.json')['files'])
    import numpy as np
    import onnx
    import onnxruntime as ort
    assert ort.__version__ == '1.29.0'
    for name in ['numpy', 'onnx', 'onnxruntime']:
        assert str(Path(sys.modules[name].__file__).resolve()) == spec['packages'][name]
    own = monitor.psutil.Process(); assert own.cpu_affinity() == [2]
    output = BASE/'native'; output.mkdir()
    captured = read(BASE/'output/result.json'); assert captured['passed'] and len(captured['calls']) == 12
    model = onnx.load(MODEL); nodes = [n for n in model.graph.node if n.op_type == 'LSTM']; assert len(nodes) == 4
    def load(item):
        path = BASE/'output'/item['file']; assert pin(path) == {k:item[k] for k in ['bytes','sha256']}
        return np.fromfile(path,dtype='<f4').reshape(item['shape'])
    def compare(actual, expected):
        assert actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        error = np.abs(actual.astype('float64')-expected.astype('float64'))/np.maximum(1.,np.abs(expected.astype('float64')))
        return dict(values=int(actual.size), failed=int(np.count_nonzero(error > 1e-4)), maximum=float(error.max(initial=0)))
    def session(model_bytes):
        options = ort.SessionOptions(); options.log_severity_level = 4
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ['session.intra_op.allow_spinning','session.inter_op.allow_spinning']: options.add_session_config_entry(key,'0')
        result = ort.InferenceSession(model_bytes,options,providers=['CPUExecutionProvider'])
        assert result.get_providers() == ['CPUExecutionProvider']; return result
    reports = []; checks = []
    for ordinal, call in enumerate(captured['calls']):
        node = nodes[call['index']]; assert call['node'] == node.name and call['opset'] == 17
        assert call['input_names'] == list(node.input) and call['output_names'] == list(node.output)
        attrs = {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attrs == dict(direction=b'bidirectional',hidden_size=128)
        assert call['attributes'] == dict(direction='bidirectional',hidden_size=128)
        inputs = [(name,item) for name,item in zip(call['input_names'],call['inputs'],strict=True) if name]
        graph = onnx.helper.make_graph([copy.deepcopy(node)], 'captured-lstm',
            [onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,item['shape']) for name,item in inputs],
            [onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,item['shape']) for name,item in zip(call['output_names'],call['outputs'],strict=True)])
        derived = onnx.helper.make_model(graph,opset_imports=[copy.deepcopy(v) for v in model.opset_import],ir_version=model.ir_version)
        assert derived.graph.node[0].SerializeToString() == node.SerializeToString(); onnx.checker.check_model(derived)
        destination = output/(str(ordinal).zfill(2)+'.onnx'); onnx.save(derived,destination)
        native = session(derived.SerializeToString()); values = {name:load(item) for name,item in inputs}; before = {n:a.tobytes() for n,a in values.items()}
        first = native.run(call['output_names'],values); held = [a.tobytes() for a in first]
        second = native.run(call['output_names'],values)
        assert all(a.tobytes() == b.tobytes() == h for a,b,h in zip(first,second,held,strict=True))
        assert all(a.tobytes() == before[n] for n,a in values.items())
        for index,(actual,item) in enumerate(zip(first,call['outputs'],strict=True)):
            comparison = compare(load(item),actual); file = output/(str(ordinal).zfill(2)+'-'+str(index)+'.f32'); actual.astype('<f4').tofile(file)
            reports.append(dict(case=call['name'],index=call['index'],node=call['node'],slot=index,comparison=comparison,
                reference=dict(file=file.name,shape=list(actual.shape),**pin(file)),derived=pin(destination),exact_repeat=True,input_unchanged=True))
    native = session(str(MODEL))
    for case in read(BASE/'capture-spec.json')['cases']:
        x = np.fromfile(case['input'],dtype='<f4').reshape(case['shape']); before=x.tobytes()
        first, = native.run(None,{'waveform':x}); held=first.tobytes(); second, = native.run(None,{'waveform':x})
        assert first.tobytes() == second.tobytes() == held and x.tobytes() == before
        selected = load(captured['tensors'][case['name']+'/scores']); retained=np.load(case['native'],allow_pickle=False)
        checks.append(dict(case=case['name'],selected=compare(selected,first),retained=compare(first,retained),exact_repeat=True,input_unchanged=True))
    assert len(reports)==36 and len(checks)==3
    passed=all(r['comparison']['failed']==0 for r in reports) and all(r[k]['failed']==0 for r in checks for k in ['selected','retained'])
    save(output/'result.json',dict(passed=passed,reports=reports,graphs=checks,version=ort.__version__,pid=own.pid,birth=own.create_time(),
        no_performance_measurement=True,capture=pin(BASE/'output/result.json'),maximum=max(r['comparison']['maximum'] for r in reports)))
    verify(spec['files']); assert passed, 'Native output bound failed; preserve complete results.'


if __name__ == '__main__': main()
