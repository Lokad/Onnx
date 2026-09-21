"""Freeze the diagnostic or execute one of its eight fresh processes."""
import argparse
import collections
import json
import shutil
import subprocess
import time
from protocol import *
import onnx


def prepare():
    assert not BASE.exists()
    assert shutil.disk_usage(ROOT).free >= LIMITS['disk']
    receipt = FULL/'closed.json'
    assert pin(receipt)['sha256'] == '24b132085567d6f84cec2582ef3910cac892533191a353d4bc25af6e5769ff0e'
    closed = read(receipt); assert closed['structural_passed'] and closed['reference_passed']
    prior = read(FULL/'manifest.json')
    assert pin(FULL/'manifest.json') == closed['files']['manifest.json']
    files = {rel(receipt): pin(receipt), rel(FULL/'manifest.json'): pin(FULL/'manifest.json')}
    for path in [MODEL, DATA, ROOT/prior['trace_model']]:
        assert pin(path) == prior['files'][rel(path)]
        files[rel(path)] = pin(path)
    assert pin(MODEL)['sha256'] == ORIGINAL and pin(DATA)['sha256'] == WEIGHTS
    requests = []; jobs = []
    for position, original in enumerate(SELECTED):
        source_job = next(j for j in prior['jobs'] if j['request'] == original and j['features'] == 'managed' and j['engine'] == 'numpy')
        desc = source_job['input']; source(desc)
        assert pin(ROOT/desc['file']) == prior['files'][desc['file']]
        files[desc['file']] = pin(ROOT/desc['file']); references = {}
        for engine in ['numpy', 'ort']:
            folder = FULL/'outputs'/f'{original:02}-managed-{engine}'
            path = folder/'result.json'
            assert pin(path) == closed['files'][path.relative_to(FULL).as_posix()]
            files[rel(path)] = pin(path); result = read(path)
            assert result['complete'] and result['input_sha256'] == desc['raw_sha256']
            references[engine] = []
            for row in result['outputs']:
                path = folder/row['file']; identity = pin(path)
                assert identity == row['pin'] == closed['files'][path.relative_to(FULL).as_posix()]
                files[rel(path)] = identity
                references[engine].append(dict(file=rel(path), shape=row['shape'], name=row['name'], dtype='<f8'))
        baselines = {}
        for kind in ['MM', 'NM']:
            item = prior['requests'][original]['baselines'][kind]
            source(item); assert pin(ROOT/item['file']) == prior['files'][item['file']]
            files[item['file']] = pin(ROOT/item['file']); baselines[kind] = item
        requests.append(dict(index=position, original=original, name=source_job['name'], input=desc, references=references, baselines=baselines))
        for mode in MODES:
            jobs.append(dict(id=f'{position:02}-{mode}', request=position, mode=mode))
    for path in list(Path(__file__).parent.glob('*.py'))+[TOOLS/'common.py', TOOLS/'interpreter.py', TOOLS/'worker.py']:
        files[rel(path)] = pin(path)
    for name, expected in prior['numerical_files'].items():
        assert pin(name) == expected, name
    model = onnx.load(MODEL, load_external_data=False)
    assert len(model.graph.node) == 1559
    BASE.mkdir()
    write(BASE/'manifest.json', dict(protocol=PROTOCOL, source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        files=files, numerical_files=prior['numerical_files'], model=prior['trace_model'], outputs=prior['outputs'], requests=requests,
        jobs=jobs, limits=LIMITS, threads=THREADS, nodes=1559, census=dict(collections.Counter(n.op_type for n in model.graph.node)),
        reference_limit=1e-4, scalar_limit=1e-7, useful_maximum_ratio=.5))
    print(json.dumps(dict(prepared=True, jobs=len(jobs), files=len(files), manifest=pin(BASE/'manifest.json'))))


def worker(job_id):
    import importlib.util
    spec = read(BASE/'manifest.json'); assert spec['protocol'] == PROTOCOL
    job = next(j for j in spec['jobs'] if j['id'] == job_id); request = spec['requests'][job['request']]
    folder = BASE/'outputs'/job_id; folder.mkdir(parents=True, exist_ok=False)
    features = source(request['input']); before = raw(features)
    model = onnx.load(ROOT/spec['model'], load_external_data=False)
    initializers = {i.name:i for i in model.graph.initializer}
    uses = collections.Counter(name for node in model.graph.node for name in node.input)
    values = {model.graph.input[0].name:features}; records = []; saved = {}; dot_checks = []
    wanted = {o['name']:(i,o) for i,o in enumerate(spec['outputs'])}; started = time.monotonic()
    for index, node in enumerate(model.graph.node):
        inputs = [values[name] if name in values else tensor_array(initializers[name], MODEL.parent) for name in node.input]
        assert all(v.dtype in [np.float32, np.int64] for v in inputs)
        result = calculate(node, inputs, job['mode']); assert result.dtype == np.float32
        if node.op_type == 'MatMul' and job['mode'] == 'wide-matmul':
            dot_checks.append(dict(node=node.name, checks=scalar_dots(inputs[0], inputs[1], result)))
        name = node.output[0]
        records.append(dict(index=index, name=node.name, op=node.op_type, dtype=str(result.dtype), shape=list(result.shape)))
        if name in wanted:
            output_index, desc = wanted[name]; assert name not in saved and list(result.shape) == desc['shape']
            path = folder/f'{output_index:02}.f32'
            with path.open('xb') as stream:
                np.ascontiguousarray(result).tofile(stream)
            saved[name] = dict(index=output_index, name=name, shape=desc['shape'], file=path.name, pin=pin(path))
            print(json.dumps(dict(boundary=output_index, seconds=time.monotonic()-started)), flush=True)
        if uses[name]:
            values[name] = result
        for name in node.input:
            uses[name] -= 1
            if uses[name] == 0:
                values.pop(name, None)
        del inputs, result
    assert len(records) == 1559 and not values and set(saved) == set(wanted) and raw(features) == before
    module_spec = importlib.util.spec_from_file_location('qualified_reference_runtime', TOOLS/'worker.py')
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    runtime = module.runtime('numpy')
    write(folder/'result.json', dict(complete=True, job=job, manifest=pin(BASE/'manifest.json'), input_unchanged=True,
        input_sha256=before, runtime=runtime, records=records, scalar_checks=dot_checks,
        outputs=[saved[o['name']] for o in spec['outputs']], seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'worker']); parser.add_argument('--job')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        worker(args.job)
