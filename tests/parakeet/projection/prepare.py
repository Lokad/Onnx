"""Freeze output-only instrumentation and reuse verified original controls."""
from common import *
import copy
import onnx
import onnxruntime as ort
import subprocess


def main():
    assert not BASE.exists()
    prior_path = ROOT/'artifacts/parakeet-stem-reference-final-20260921/verified.json'
    assert pin(prior_path)['sha256'] == 'ad5ed0f1876ca519230ba754274e6b2f6a020dce614b5b45b36c92234f2c0b22'
    prior = read(prior_path)
    for path, expected in prior['files'].items(): assert pin(ROOT/path) == expected, path
    assert all(absent(i) for i in prior['identities'])
    trace_receipt = ROOT/'artifacts/parakeet-layer-trace-final-20260921/verified.json'
    assert pin(trace_receipt)['sha256'] == '26aeecede5769a5cc1f96774a92e928427dafabd1a2deee312c9e5b683eff025'
    trace_files = read(trace_receipt)['files']
    old = read(TRACE/'manifest.json'); reference = read(REFERENCE/'manifest.json')
    assert np.__version__ == old['numpy'] == reference['numpy'] and ort.__version__ == old['onnxruntime']
    model_path = ROOT/old['models']['plain']
    assert pin(model_path) == old['files'][rel(model_path)]
    model = onnx.load(model_path, load_external_data=False); trace = copy.deepcopy(model)
    for name in (STEM, RESHAPE):
        trace.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [None, None, None]))
    stripped = copy.deepcopy(trace); del stripped.graph.output[2:]
    assert stripped.SerializeToString() == model.SerializeToString()
    BASE.mkdir(); (BASE/'models').mkdir()
    target = BASE/'models/encoder-trace.onnx'; onnx.save_model(trace, target)
    files = {}; worker_files = {}
    def bind(path, worker=False):
        name = rel(path); files[name] = pin(path)
        if worker: worker_files[name] = files[name]
        return name
    bind(prior_path); bind(trace_receipt); bind(TRACE/'manifest.json'); bind(REFERENCE/'manifest.json'); bind(model_path)
    bind(target, True)
    for name in {e.value for t in model.graph.initializer for e in t.external_data if e.key == 'location'}:
        assert Path(name).name == name
        source = model_path.parent/name
        assert pin(source) == old['files'][rel(source)]
        os.link(source, target.parent/name); bind(target.parent/name, True)
    for route in old['inputs'].values():
        for name in route.values():
            assert pin(ROOT/name) == old['files'][name]; bind(ROOT/name, True)
    for path in (TRACE/'bin').iterdir():
        assert pin(path) == old['files'][rel(path)]; bind(path, True)
    for name, expected in old['files'].items():
        if 'site-packages/onnxruntime/' in name:
            assert pin(ROOT/name) == expected; bind(ROOT/name, True)
    controls = {}; references = {}
    for job in CAPTURES[:4]:
        route, engine, kind = job['id'], job['engine'], job['input']
        folder = TRACE/'outputs'/('managed-'+kind+'-trace') if engine == 'managed' else DYNAMIC/'outputs'/kind
        assert pin(folder/'result.json') == trace_files[rel(folder/'result.json')]
        result = read(folder/'result.json'); bind(folder/'result.json'); rows = {}
        for record in result['outputs']:
            if record['name'] in OUTPUTS:
                array(folder/record['file'], record)
                assert pin(folder/record['file']) == trace_files[rel(folder/record['file'])]
                rows[record['name']] = dict(record, file=bind(folder/record['file']))
        assert set(rows) == set(OUTPUTS[:3]); controls[route] = rows
    managed_nodes = TRACE/'outputs/managed-native-plain/nodes.json'
    native_nodes = ROOT/'artifacts/parakeet-trace-optimization-20260921/plain/nodes.json'
    for path in (managed_nodes, native_nodes): assert pin(path) == trace_files[rel(path)]
    bind(managed_nodes); bind(native_nodes)
    for kind in ('native', 'managed'):
        for engine in ('numpy', 'torch'):
            folder = REFERENCE/'outputs'/(engine+'-'+kind); r = read(folder/'result.json'); bind(folder/'result.json')
            rows = {}
            for record in r['outputs']:
                if record['name'] in ('reshape', 'projection', 'stem'):
                    array(folder/record['file'], record)
                    rows[record['name']] = dict(record, file=bind(folder/record['file']))
            references[engine+'-'+kind] = rows
    weights = {k:v for k,v in reference['weights'].items() if k.startswith('projection.')}
    for value in weights.values(): bind(ROOT/value['file'], True)
    for name, expected in reference['numerical_libraries'].items():
        assert pin(name) == expected; bind(name)
    for path in Path(__file__).parent.iterdir():
        if path.is_file(): bind(path, True)
    bind(ROOT/'tests/pyannote/filterbank-reference/common.py')
    bind(ROOT/'tests/parakeet/layer-trace/Program.cs')
    write(BASE/'manifest.json', dict(protocol=PROTOCOL, diagnostic=DIAGNOSTIC, base=rel(BASE),
        jobs=JOBS, limits=LIMITS, models=dict(trace=rel(target)), outputs=dict(trace=OUTPUTS),
        inputs=old['inputs'], controls=controls, references=references, weights=weights,
        nodes=dict(managed=rel(managed_nodes), native=rel(native_nodes)), files=files, worker_files=worker_files,
        numerical_libraries=reference['numerical_libraries'], numpy=np.__version__, torch=reference['torch'],
        onnxruntime=ort.__version__, source=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()))
    print(json.dumps(dict(manifest=pin(BASE/'manifest.json'), captures=len(CAPTURES), reference_workers=2)))


if __name__ == '__main__': main()
