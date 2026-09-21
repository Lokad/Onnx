"""One fresh local reference worker with all twelve layer outputs retained."""
import argparse
import importlib.util
import json
import time
import onnx
from protocol import *


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--job', required=True); args = parser.parse_args()
    spec = read(BASE/'manifest.json'); job = next(j for j in spec['jobs'] if j['id'] == args.job)
    assert spec['protocol'] == PROTOCOL
    folder = BASE/'outputs'/job['id']; folder.mkdir(parents=True, exist_ok=False)
    x = load(job['input']); before = raw(x); saved = {}; wanted = {o['name']: (i, o) for i, o in enumerate(spec['outputs'])}
    started = time.monotonic()
    def capture(name, value):
        if name not in wanted:
            return
        assert name not in saved
        index, desc = wanted[name]
        assert value.dtype == np.float64 and list(value.shape) == desc['shape'] and np.isfinite(value).all()
        path = folder/f'{index:02}.f64'
        with path.open('xb') as stream:
            np.ascontiguousarray(value, dtype='<f8').tofile(stream)
        saved[name] = dict(index=index, name=name, shape=desc['shape'], file=path.name, pin=pin(path))
    if job['engine'] == 'numpy':
        from interpreter import run
        records = run(onnx.load(MODEL, load_external_data=False), MODEL.parent, x, capture)
        assert len(records) == 48
    else:
        from native import run
        directory = ROOT/spec['promoted']
        records = run(directory, read(directory/'stages.json'), '/layers.19/Add_1_output_0', x, capture)
        assert len(records) == 3
    assert set(saved) == set(wanted) and raw(x) == before
    module_spec = importlib.util.spec_from_file_location('qualified_full_reference_worker', REFERENCE_TOOLS/'worker.py')
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    write(folder/'result.json', dict(complete=True, job=job, manifest=pin(BASE/'manifest.json'), input_unchanged=True,
          input_sha256=before, runtime=module.runtime(job['engine']), seconds=time.monotonic()-started,
          outputs=[saved[o['name']] for o in spec['outputs']], records=records))
    print(json.dumps(dict(job=job['id'], complete=True, arrays=len(saved))))


if __name__ == '__main__':
    main()
