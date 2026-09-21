"""Native capture or an independent float64 projection on all actual inputs."""
from common import *
import math
import re


def native(spec, job, folder):
    import onnxruntime as ort
    assert ort.__version__ == spec['onnxruntime']
    inputs = spec['inputs'][job['input']]
    feeds = dict(audio_signal=np.load(ROOT/inputs['features'], allow_pickle=False),
                 length=np.load(ROOT/inputs['length'], allow_pickle=False))
    before = {k:v.tobytes() for k,v in feeds.items()}
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
        options.add_session_config_entry(key, '0')
    if job['id'] == 'native-native':
        options.optimized_model_filepath = str(folder/'optimized.onnx')
        options.add_session_config_entry('session.optimized_model_external_initializers_file_name', 'weights.bin')
        options.add_session_config_entry('session.optimized_model_external_initializers_min_size_in_bytes', '1024')
    inference = ort.InferenceSession(str(ROOT/spec['models']['trace']), options, providers=['CPUExecutionProvider'])
    assert [v.name for v in inference.get_outputs()] == OUTPUTS
    values = inference.run(None, feeds); held = [v.tobytes() for v in values]
    assert all(v.tobytes() == before[k] for k,v in feeds.items()); del inference
    assert all(v.tobytes() == bits for v,bits in zip(values, held, strict=True))
    records = []
    for i,(name,value) in enumerate(zip(OUTPUTS, values, strict=True)):
        r = output(folder, f'{i:02}', value); r['name'] = name; records.append(r)
    write(folder/'result.json', dict(complete=True, job=job, outputs=records, manifest=pin(BASE/'manifest.json'),
        inputs_unchanged=True, held_outputs_unchanged=True, onnxruntime=ort.__version__, affinity=[2],
        settings=dict(threads=1, execution='sequential', optimization='all', spinning=False)))


def reference(spec, job, folder):
    engine = job['engine']; torch_config = None
    if engine == 'torch':
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == spec['torch'] and torch.get_num_threads() == torch.get_num_interop_threads() == 1
        torch_config = dict(parallel=torch.__config__.parallel_info(), build=torch.__config__.show())
        assert re.search(r'mkl_get_max_threads\(\)\s*:\s*1\b', torch_config['parallel']) and 'BLAS_INFO=mkl' in torch_config['build']
    w = np.load(ROOT/spec['weights']['projection.weight']['file'], allow_pickle=False)
    b = np.load(ROOT/spec['weights']['projection.bias']['file'], allow_pickle=False)
    assert w.dtype == b.dtype == np.float32 and w.shape == (4096,1024) and b.shape == (1024,)
    weight_before = (w.tobytes(), b.tobytes()); wd = w.astype(np.float64); bd = b.astype(np.float64)
    rows = []; probes = []; held = []
    for route in ROUTES:
        x = tensors(BASE/'outputs'/route)[RESHAPE]
        assert x.dtype == np.float32 and x.shape == (1,74,4096)
        before = x.tobytes(); xd = x.astype(np.float64)
        if engine == 'numpy': projection = xd @ wd
        else: projection = torch.matmul(torch.from_numpy(xd), torch.from_numpy(wd)).numpy().copy()
        stem = projection + bd
        sub = folder/route; sub.mkdir()
        records = [output(sub, name, value) for name,value in [('projection',projection), ('stem',stem)]]
        for index in coordinates():
            row,col = divmod(index,1024)
            expected = math.fsum(float(x[0,row,k])*float(w[k,col]) for k in range(4096))
            actual = float(projection.flat[index]); error = abs(actual-expected)/max(1.,abs(expected))
            assert error <= REFERENCE_LIMIT
            probes.append(dict(route=route, index=index, expected=expected, actual=actual, error=error))
        assert x.tobytes() == before
        held.extend((v, v.tobytes()) for v in (projection,stem))
        write(sub/'result.json', dict(complete=True, outputs=records))
        rows.append(dict(route=route, result=rel(sub/'result.json')))
    assert (w.tobytes(),b.tobytes()) == weight_before and all(v.tobytes() == bits for v,bits in held)
    h = helpers(); assert h.openblas_threads() == 1
    libraries = h.libraries(psutil.Process())
    for path,expected in libraries.items(): assert spec['numerical_libraries'].get(path) == expected, path
    assert not any('onnxruntime' in m.path.lower() for m in psutil.Process().memory_maps())
    write(folder/'result.json', dict(complete=True, job=job, manifest=pin(BASE/'manifest.json'), rows=rows,
        probes=probes, inputs_unchanged=True, weights_unchanged=True, held_outputs_unchanged=True,
        numpy=np.__version__, libraries=libraries, torch_config=torch_config, openblas_threads=1, native_ort_loaded=False))


def main():
    spec = read(BASE/'manifest.json'); verify(spec)
    assert np.__version__ == spec['numpy'] and psutil.Process().cpu_affinity() == [2]
    assert not any(k.lower().startswith(('lokad_', 'dotnet_', 'complus_')) for k in os.environ)
    job = next(j for j in JOBS if j['id'] == sys.argv[1]); assert job['engine'] != 'managed'
    folder = BASE/'outputs'/job['id']; folder.mkdir(parents=True, exist_ok=False)
    (native if job['engine'] == 'native' else reference)(spec, job, folder)
    print(json.dumps(dict(complete=True, job=job['id'])))


if __name__ == '__main__': main()
