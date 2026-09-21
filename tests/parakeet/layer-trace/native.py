"""One native encoder request or the declared fixed-state decoder comparison."""
from common import *
import numpy as np
import onnxruntime as ort
import psutil


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
        options.add_session_config_entry(key, '0')
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def outputs(folder, names, values):
    result = []
    for index,(name,value) in enumerate(zip(names, values, strict=True)):
        assert value.dtype in (np.float32, np.int32, np.int64) and np.isfinite(value).all()
        path = folder/f'{index:02}.bin'
        with path.open('xb') as stream:
            stream.write(value.tobytes(order='C'))
        result.append(dict(name=name, file=path.name, shape=list(value.shape), dtype=str(value.dtype), **pin(path)))
    return result


def main():
    spec = read(BASE/'manifest.json'); verify(spec)
    assert ort.__version__ == spec['onnxruntime'] and np.__version__ == spec['numpy']
    assert psutil.Process().cpu_affinity() == [2]
    assert not any(k.lower().startswith(('lokad_', 'dotnet_', 'complus_')) for k in os.environ)
    identifier = sys.argv[1]; folder = BASE/'outputs'/identifier; folder.mkdir(parents=True, exist_ok=False)
    if identifier == 'decode':
        decoder = spec['decoder']; inference = session(ROOT/decoder['model']); rows = []; held = []
        state = [np.load(ROOT/p, allow_pickle=False) for p in decoder['states']]
        for job in spec['jobs']:
            prior = BASE/'outputs'/job['id']; result = read(prior/'result.json'); assert result['complete']
            record = next(r for r in result['outputs'] if r['name'] == 'outputs')
            encoded = array(prior/record['file'], record)
            assert encoded.shape == (1, 1024, 74)
            frame = decoder['frame']
            feeds = dict(encoder_outputs=np.ascontiguousarray(encoded[:, :, frame:frame+1]),
                         targets=np.array([[decoder['target']]], np.int32), target_length=np.array([1], np.int32),
                         input_states_1=state[0], input_states_2=state[1])
            before = {k:v.tobytes() for k,v in feeds.items()}
            values = inference.run(None, feeds)
            assert all(v.tobytes() == before[k] for k,v in feeds.items())
            assert all(v.tobytes() == b for v,b in held)
            held.extend((v, v.tobytes()) for v in values)
            target = folder/job['id']; target.mkdir()
            rows.append(dict(job=job, outputs=outputs(target, [v.name for v in inference.get_outputs()], values)))
        write(folder/'result.json', dict(complete=True, rows=rows, manifest_sha256=pin(BASE/'manifest.json')['sha256'],
                                        inputs_unchanged=True, held_outputs_unchanged=True, affinity=[2], onnxruntime=ort.__version__))
        print('decode:', len(rows), 'fixed-state calls', flush=True)
        return
    job = next(j for j in spec['jobs'] if j['id'] == identifier); assert job['engine'] == 'native'
    route = spec['inputs'][job['input']]
    feeds = dict(audio_signal=np.load(ROOT/route['features'], allow_pickle=False), length=np.load(ROOT/route['length'], allow_pickle=False))
    before = {k:v.tobytes() for k,v in feeds.items()}
    inference = session(ROOT/spec['models'][job['mode']]); names = [v.name for v in inference.get_outputs()]
    assert names == spec['outputs'][job['mode']]
    values = inference.run(None, feeds)
    held = [v.tobytes() for v in values]
    assert all(v.tobytes() == before[k] for k,v in feeds.items())
    del inference
    assert all(v.tobytes() == b for v,b in zip(values, held, strict=True))
    write(folder/'result.json', dict(complete=True, job=job, outputs=outputs(folder, names, values),
          manifest_sha256=pin(BASE/'manifest.json')['sha256'], inputs_unchanged=True, held_outputs_unchanged=True,
          affinity=[2], onnxruntime=ort.__version__, numpy=np.__version__, settings=dict(threads=1, execution='sequential', optimization='all', spinning=False)))
    print(identifier, len(names), 'arrays', flush=True)


if __name__ == '__main__':
    main()
