"""Verify original evidence and freeze the fixed twelve-worker encoder schedule."""
from common import *
import copy
import shutil
import subprocess
import numpy as np
import onnx
import onnxruntime as ort


def main():
    assert not BASE.exists(), 'Use a new evidence directory'
    assert np.__version__ == '2.2.4' and ort.__version__ == '1.29.0'
    assert pin(PRODUCT/'Lokad.Onnx.dll')['sha256'] == CORE
    BASE.mkdir(); inputs = BASE/'inputs'; inputs.mkdir(); models = BASE/'models'; models.mkdir()
    old_manifest = read(OLD/'reference/manifest.json'); managed = read(OLD/'default.json')
    case = old_manifest['cases'][0]; actual = managed['rows'][0]
    assert case['name'] == actual['name'] == 'english-16k'
    assert case['steps'][26]['frame'] == 49 and case['steps'][26]['target'] == 1627
    files = {}
    def bind(path):
        files[rel(path)] = pin(path)
        return rel(path)
    bind(OLD/'reference/manifest.json'); bind(OLD/'default.json'); bind(OLD/'frontend-probe/result.json')
    def native(name):
        path = OLD/'reference'/name; expected = old_manifest['files'][name]
        assert pin(path) == {k:expected[k] for k in ('bytes', 'sha256')}
        bind(path); return path
    def own(label, name):
        item = next(v for v in actual['comparisons'] if v['label'] == label and v['output'] == name)
        path = OLD/'default.json.tensors'/item['file']
        assert pin(path)['sha256'] == item['sha256']; bind(path)
        dtype = {'Float':'float32', 'Int32':'int32', 'Int64':'int64'}[item['dtype']]
        value = np.fromfile(path, dtype=dtype).reshape(item['shape']); assert np.isfinite(value).all()
        target = inputs/(label+'-'+name+'.npy'); np.save(target, value, allow_pickle=False); bind(target)
        return target
    routes = {}
    for kind in ('native', 'managed'):
        routes[kind] = {}
        for key, output in [('features', 'features'), ('length', 'features_lens')]:
            path = native(case['stages'][0]['outputs'][output]) if kind == 'native' else own('frontend', output)
            routes[kind][key] = bind(path)
    controls = {'native-native':bind(native(case['stages'][1]['outputs']['outputs'])),
                'managed-managed':bind(own('encoder', 'outputs'))}
    substitution = OLD/'frontend-probe/managed-features-encoded.npy'
    assert pin(substitution)['sha256'] == read(OLD/'frontend-probe/result.json')['outputs'][substitution.name]
    controls['native-managed'] = bind(substitution)
    decoder = dict(frame=49, target=1627,
                   states=[bind(native(case['steps'][25]['outputs']['output_states_'+str(i)])) for i in (1, 2)],
                   expected={k:bind(native(v)) for k,v in case['steps'][26]['outputs'].items()})
    original = ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    assets = old_manifest['assets']['files']
    for name in ('encoder-model.onnx', 'encoder-model.onnx.data', 'decoder_joint-model.onnx'):
        path = original.parent/name
        assert pin(path) == assets[name], name
        bind(path)
    for path in original.parent.glob('decoder_joint-model.onnx*'):
        if path.name in assets:
            assert pin(path) == assets[path.name]; bind(path)
    decoder['model'] = rel(original.parent/'decoder_joint-model.onnx')
    model = onnx.load(original, load_external_data=False); trace = copy.deepcopy(model)
    selected = ['/pre_encode/out/Add_output_0'] + [f'/layers.{i}/norm_out/LayerNormalization_output_0' for i in range(24)]
    producers = {v:n for n in model.graph.node for v in n.output}
    assert len(model.graph.node) == 4491 and all(v in producers for v in selected)
    for name in selected:
        trace.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 74, 1024]))
    stripped = copy.deepcopy(trace); del stripped.graph.output[len(model.graph.output):]
    assert stripped.SerializeToString() == model.SerializeToString(), 'Only graph outputs may change'
    target = models/'encoder-trace.onnx'; onnx.save_model(trace, target)
    for name in {e.value for t in model.graph.initializer for e in t.external_data if e.key == 'location'}:
        assert Path(name).name == name
        os.link(original.parent/name, models/name); bind(models/name)
    bind(target)
    build = ['dotnet', 'build', 'tests/parakeet/layer-trace/Trace.csproj', '-c', 'Release', '--tl:off', '--nologo', '-v', 'minimal',
             '-p:FrozenProductDirectory='+str(PRODUCT), '-o', str(BASE/'bin')]
    with (BASE/'build.log').open('x') as log:
        subprocess.run(build, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert pin(BASE/'bin/Lokad.Onnx.dll')['sha256'] == CORE
    for path in sorted((BASE/'bin').iterdir()):
        if path.is_file(): bind(path)
    for path in sorted(Path(__file__).parent.glob('*')):
        if path.is_file(): bind(path)
    bind(ROOT/'tests/Shared/NpySupport.cs')
    for path in sorted((SITE/'onnxruntime/capi').iterdir()):
        if path.suffix in ('.dll', '.pyd'): bind(path)
    jobs = [dict(id=f'{engine}-{kind}-{mode}{"-repeat" if repeat else ""}', engine=engine, input=kind, mode=mode, repeat=repeat)
            for engine in ('managed', 'native') for kind in ('native', 'managed')
            for mode,repeat in [('plain', False), ('trace', False), ('trace', True)]]
    worker_files = dict(files)
    spec = dict(protocol=PROTOCOL, base=rel(BASE), limits=LIMITS, core=CORE, models=dict(plain=rel(original), trace=rel(target)),
                outputs=dict(plain=[v.name for v in model.graph.output], trace=[v.name for v in trace.graph.output]),
                inputs=routes, controls=controls, decoder=decoder, jobs=jobs, files=files, worker_files=worker_files,
                numpy=np.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__, source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(BASE/'manifest.json', spec)
    print(json.dumps(dict(manifest=pin(BASE/'manifest.json'), jobs=len(jobs), outputs=len(trace.graph.output))))


if __name__ == '__main__':
    main()
