"""Bind saved layer-20 inputs and reuse unchanged promoted reference weights."""
import copy
import os
import shutil
import subprocess
import onnx
from protocol import *
from promote import partition


def main():
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    assert not BASE.exists() and shutil.disk_usage(ROOT).free >= LIMITS['disk']
    receipts = {}
    for directory, filename, sha in [
        (PRIOR, 'closed-v2.json', '13478ae21b6a545639cac4ad27f7465aafa46a010aa9d56f9d802745a40ebe32'),
        (FULL, 'closed.json', '24b132085567d6f84cec2582ef3910cac892533191a353d4bc25af6e5769ff0e')]:
        assert pin(directory/filename)['sha256'] == sha
        receipts[directory] = read(directory/filename)
    files = {}
    def bind(path, expected=None):
        actual = pin(path)
        if expected is not None:
            assert actual == expected, path
        files[rel(path)] = actual
        return actual
    def closed(directory, name):
        path = directory/name; bind(path, receipts[directory]['files'][name]); return path
    for directory, filename in [(PRIOR, 'closed-v2.json'), (FULL, 'closed.json')]:
        bind(directory/filename)
    prior = read(closed(PRIOR, 'manifest.json')); full = read(closed(FULL, 'manifest.json'))
    assert receipts[PRIOR]['passed'] is True and receipts[FULL]['reference_passed'] is True
    assert all(absent(b) for b in receipts[PRIOR]['births'])
    assert pin(MODEL)['sha256'] == 'bd03b8953354ca3d9fa5dda33c4f9c76e15bbd56af9ae0e92559633b9e9ef6fd'
    bind(MODEL, prior['files'][rel(MODEL)])
    original = onnx.load(MODEL, load_external_data=False)
    promoted_path = closed(FULL, 'promoted/model.onnx')
    promoted = onnx.load(promoted_path, load_external_data=False)
    nodes = {n.name: n for n in promoted.graph.node}
    assert len(original.graph.node) == 48 and len(original.graph.initializer) == 27
    assert all(n.SerializeToString() == nodes[n.name].SerializeToString() for n in original.graph.node)
    double = copy.deepcopy(original); constants = {i.name: i for i in promoted.graph.initializer}
    del double.graph.initializer[:]
    for item in original.graph.initializer:
        upgraded = constants[item.name]
        a = tensor_array(item, MODEL.parent); b = tensor_array(upgraded, promoted_path.parent)
        assert np.array_equal(a.astype(np.float64) if a.dtype == np.float32 else a, b), item.name
        double.graph.initializer.append(upgraded)
    for value in list(double.graph.input)+list(double.graph.output)+list(double.graph.value_info):
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            value.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    double.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    full_jobs = {(j['request'], j['features'], j['engine']): j for j in full['jobs']}
    requests = []
    for request in prior['requests']:
        entry = dict(request=request['request'], selected_request=request['selected_request'],
                     original_request=request['original_request'], name=request['name'], features=request['features'], full_final={}, actual={})
        for engine in ['managed', 'native']:
            desc = request[engine+'_input']; identity = bind(ROOT/desc['file'], prior['files'][desc['file']])
            assert identity['sha256'] == desc['raw_sha256'] and desc['shape'] == [1, 1500, 1280]
            entry[engine+'_input'] = dict(file=desc['file'], pin=identity, dtype='<f4', shape=desc['shape'])
        for engine in ['numpy', 'ort']:
            job = full_jobs[request['original_request'], request['features'], engine]
            folder = FULL/'outputs'/job['id']; result = read(closed(FULL, 'outputs/'+job['id']+'/result.json'))
            assert result['complete'] is True and job['name'] == request['name']
            def saved(index):
                row = result['outputs'][index]; path = closed(FULL, 'outputs/'+job['id']+'/'+row['file'])
                assert pin(path) == row['pin'] and row['shape'] == [1, 1500, 1280]
                return dict(file=rel(path), pin=row['pin'], dtype='<f8', shape=row['shape'])
            assert result['outputs'][27]['name'] == original.graph.input[0].name
            assert result['outputs'][28]['name'] == prior['outputs'][11]['name']
            if engine == 'numpy':
                entry['reference_input'] = saved(27)
            entry['full_final'][engine] = saved(28)
        for job in [j for j in prior['schedule'] if j['request'] == request['request']]:
            folder = PRIOR/'outputs'/job['id']; result = read(closed(PRIOR, 'outputs/'+job['id']+'/result.json'))
            for record in result['records']:
                assert record['kind'] not in entry['actual']
                outputs = []
                for index, row in enumerate(record['outputs']):
                    assert row['name'] == prior['outputs'][index]['name'] and row['shape'] == prior['outputs'][index]['shape']
                    path = closed(PRIOR, 'outputs/'+job['id']+'/'+row['file'])
                    identity = pin(path); assert identity['sha256'] == row['sha256']
                    outputs.append(dict(file=rel(path), pin=identity, dtype='<f4', shape=row['shape']))
                entry['actual'][record['kind']] = outputs
        assert set(entry['actual']) == {'MM', 'MN', 'NM', 'NN'}
        requests.append(entry)
    assert len(requests) == 8 and [r['request'] for r in requests] == list(range(8))
    # Bind only files used by this new computation, against the prior receipts.
    bind(MODEL.parent/'encoder_model.onnx_data', prior['files'][rel(MODEL.parent/'encoder_model.onnx_data')])
    weight = closed(FULL, 'promoted/weights.f64')
    for name in ['common.py', 'interpreter.py', 'native.py', 'promote.py', 'worker.py']:
        path = REFERENCE_TOOLS/name; bind(path, full['files'][rel(path)])
    for name, wanted in full['numerical_files'].items():
        assert pin(name) == wanted, name
    BASE.mkdir(); destination = BASE/'promoted'; destination.mkdir()
    os.link(weight, destination/'weights.f64')
    onnx.save_model(double, destination/'model.onnx')
    stages = partition(double, destination); assert len(stages) == 3 and stages[1]['kind'] == 'math_erf'
    write(destination/'stages.json', stages)
    for path in destination.iterdir():
        bind(path)
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            bind(path)
    plan = ROOT/'.agent/m4-whisper-layer20-reference-20260921.md'
    shutil.copyfile(plan, BASE/'prospective-plan.md'); bind(BASE/'prospective-plan.md')
    jobs = [dict(id=f"r{r['request']:02}-{incoming}-{engine}", request=r['request'], incoming=incoming, engine=engine,
                 input=r[incoming+'_input']) for r in requests for incoming in ['reference', 'managed', 'native'] for engine in ['numpy', 'ort']]
    meta = dict(protocol=PROTOCOL, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                files=files, numerical_files=full['numerical_files'], outputs=prior['outputs'], requests=requests, jobs=jobs,
                original=rel(MODEL), promoted=rel(destination), limits=LIMITS, threads=THREADS, reference_limit=1e-9, original_limit=1e-4,
                output_bytes=48*sum(int(np.prod(o['shape']))*8 for o in prior['outputs']))
    assert len(jobs) == 48
    write(BASE/'manifest.json', meta)
    print(reference.json.dumps(dict(jobs=len(jobs), arrays=48*12, bytes=meta['output_bytes'], files=len(files), manifest=pin(BASE/'manifest.json'))))


if __name__ == '__main__':
    main()
