"""Separate fresh processes for native graph construction and diagnostic inference."""
import os
from pathlib import Path
import sys
from protocol import MODELS, compare, pin, read, save, verify


def main():
    assert len(sys.argv) == 3 and sys.argv[2] in ['metadata','profile']
    base = Path(sys.argv[1]).resolve(); mode = sys.argv[2]
    assert sys.platform == 'linux' and sorted(os.sched_getaffinity(0)) == [2]
    spec = verify(base)
    import numpy as np
    import onnxruntime as ort
    import psutil
    versions = dict(onnxruntime=ort.__version__, numpy=np.__version__)
    assert ort.__version__ == '1.29.0'
    packages = {name: str(Path(module.__file__).resolve()) for name,module in [('numpy',np),('onnxruntime',ort)]}
    for path in packages.values(): assert path in spec['external'] and pin(path) == spec['external'][path]
    output = base/mode; output.mkdir()
    results = {}; held = []; inputs = []
    for model in MODELS:
        options = ort.SessionOptions(); options.log_severity_level = 4
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ['session.intra_op.allow_spinning','session.inter_op.allow_spinning']: options.add_session_config_entry(key,'0')
        options.optimized_model_filepath = str(output/(model+'-optimized.onnx'))
        options.enable_profiling = mode == 'profile'
        options.profile_file_prefix = str(output/(model+'-events'))
        session = ort.InferenceSession(spec['models'][model], options, providers=['CPUExecutionProvider'])
        assert session.get_providers() == ['CPUExecutionProvider']
        assert len(session.get_inputs()) == len(session.get_outputs()) == 1
        result = dict(original=pin(Path(spec['models'][model])), optimized=pin(output/(model+'-optimized.onnx')),
            inputs=[dict(name=v.name,shape=v.shape,type=v.type) for v in session.get_inputs()], records=[])
        if mode == 'profile':
            previous = {}
            for repeat in range(2):
                for case in [c for c in spec['cases'] if c['model'] == model]:
                    source = base/case['input']; x = np.fromfile(source,dtype='<f4').reshape(case['shape'])
                    assert pin(source) == case['input_pin'] and np.isfinite(x).all()
                    before = x.tobytes(); inputs.append((x,before))
                    actual, = session.run(None,{session.get_inputs()[0].name:x})
                    expected = np.load(base/case['reference'],allow_pickle=False)
                    check = compare(actual,expected); assert check['failed_values'] == 0
                    if case['name'] in previous: assert actual.tobytes() == previous[case['name']]
                    previous[case['name']] = actual.tobytes(); held.append((actual,actual.tobytes()))
                    target = output/(case['name']+'-'+model+'-'+str(repeat)+'.npy')
                    np.save(target,actual,allow_pickle=False)
                    result['records'].append(dict(name=case['name'],model=model,repeat=repeat,
                        input_pin=pin(source),output=target.name,output_pin=pin(target),comparison=check))
            trace = Path(session.end_profiling()); assert trace.parent == output
            events = read(trace)
            assert len([e for e in events if e.get('name') == 'model_run' and e.get('cat') == 'Session']) == 6
            result.update(profile=trace.name, profile_pin=pin(trace), events=len(events))
        results[model] = result
        del session
    assert all(x.tobytes() == before for x,before in inputs + held)
    verify(base)
    own = psutil.Process()
    save(output/'result.json',dict(passed=True,mode=mode,versions=versions,package_origins=packages,
        interpreter=pin(Path(sys.executable)),pid=own.pid,birth=own.create_time(),affinity=own.cpu_affinity(),
        settings=dict(provider='CPUExecutionProvider',intra_threads=1,inter_threads=1,sequential=True,
                      graph_optimizations='all',spinning=False,profiling=mode=='profile'),
        models=results,inputs_and_held_outputs_unchanged=True,arrays=len(held)))


if __name__ == '__main__': main()
