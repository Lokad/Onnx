"""Complete new managed-table reference, using the unchanged qualified algorithms."""
import argparse, re, time
from shared import *
from routes import numpy_route, torch_route

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); p.add_argument('--engine', choices=['numpy', 'torch'], required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); output = base / a.engine; output.mkdir(exist_ok=False)
    ps = psutil_module(); process = ps.Process(); assert process.cpu_affinity() == [0]
    spec = read(base / 'manifest.json'); verify(spec['files'])
    torch_version = None
    torch_parallel = None
    torch_config = None
    if a.engine == 'torch':
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        assert torch.__version__ == '2.11.0+cpu' and torch.get_num_threads() == torch.get_num_interop_threads() == 1
        torch_version = torch.__version__
        torch_parallel = torch.__config__.parallel_info()
        torch_config = torch.__config__.show()
        assert re.search(r'mkl_get_max_threads\(\)\s*:\s*1\b', torch_parallel)
        assert 'BLAS_INFO=mkl' in torch_config
    assert np.__version__ == '2.2.4'
    window = np.load(ROOT / spec['window']); mel = np.load(ROOT / spec['mel'])
    window_before, mel_before = window.copy(), mel.copy()
    route = numpy_route if a.engine == 'numpy' else torch_route
    records = []
    for case in spec['cases']:
        samples = np.load(ROOT / case['input'], allow_pickle=False); before = samples.copy()
        start = time.perf_counter(); stages = route(samples, window, mel); seconds = time.perf_counter() - start
        assert np.array_equal(samples, before) and np.array_equal(window, window_before) and np.array_equal(mel, mel_before)
        directory = output / case['name']; directory.mkdir(); entries = {}
        for name in STAGES:
            value = stages[name]; assert value.dtype == np.float64 and np.isfinite(value).all()
            path = directory / (name + '.npy')
            with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)
            entries[name] = dict(shape=list(value.shape), pin=pin(path))
        records.append(dict(name=case['name'], seconds=seconds, stages=entries, input_unchanged=True, coefficients_unchanged=True))
        print(case['name'], flush=True)
    blas_threads = openblas_threads(); assert blas_threads == 1
    maps = [row.path for row in process.memory_maps()]
    write(output / 'result.json', dict(complete=True, engine=a.engine, manifest=pin(base / 'manifest.json'), records=records,
          runtime=dict(pid=process.pid, birth=process.create_time(), affinity=process.cpu_affinity(), numpy=np.__version__, torch=torch_version,
                       libraries=libraries(process), blas_threads=blas_threads, torch_parallel=torch_parallel, torch_config=torch_config,
                       native_ort_loaded=any('onnxruntime' in v.lower() for v in maps))))

if __name__ == '__main__': main()
