"""Freeze an output-only managed GELU intervention against retained references."""
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
sys.path.append(str(SITE))
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
import onnx
import psutil

BASE = ROOT/'artifacts/whisper-managed-gelu-20260921'
PRIOR = ROOT/'artifacts/whisper-matmul-precision-20260921'
TRACE = ROOT/'artifacts/whisper-trace-selected-20260920'
PRODUCT = ROOT/'artifacts/whisper-memory-product-v2-20260921/source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
CORE = 'd1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
PROTOCOL = 'whisper-managed-gelu-v1'
LIMITS = dict(seconds=1800, rss=8*1024**3, available=1024**3, preflight_available=10*1024**3, disk=20*1024**3)


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def absent(identity):
    try:
        return psutil.Process(identity['pid']).create_time() != identity['birth']
    except psutil.NoSuchProcess:
        return True


def append_erf_outputs(model):
    """Change only observability, retaining exact original arithmetic/storage."""
    result = copy.deepcopy(model)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = {v.name:v for v in [*inferred.graph.value_info, *inferred.graph.output]}
    outputs = {v.name for v in result.graph.output}
    erfs = [n for n in model.graph.node if n.op_type == 'Erf']
    assert len(erfs) == 34
    for node in erfs:
        assert len(node.output) == 1 and node.output[0] not in outputs
        value = shapes[node.output[0]]
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        result.graph.output.append(value)
        outputs.add(value.name)
    restored = copy.deepcopy(result)
    del restored.graph.output[:]
    restored.graph.output.extend(model.graph.output)
    assert restored.SerializeToString(deterministic=True) == model.SerializeToString(deterministic=True)
    return result


def output_specs(model):
    outputs = []
    for value in model.graph.output:
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        dimensions = value.type.tensor_type.shape.dim
        assert all(d.HasField('dim_value') or d.dim_param == 'batch_size' for d in dimensions), value
        outputs.append(dict(name=value.name, shape=[d.dim_value if d.HasField('dim_value') else 1 for d in dimensions]))
    return outputs


def prepare():
    assert not (BASE/'manifest.json').exists()
    assert shutil.disk_usage(ROOT).free >= LIMITS['disk']
    assert pin(PRIOR/'closed.json')['sha256'] == '6baa0d3774515baf452a538e553b556a83d6c486f10b43895ab02b775a2baee9'
    prior = read(PRIOR/'manifest.json'); closed = read(PRIOR/'closed.json')
    assert pin(PRIOR/'manifest.json') == closed['files']['manifest.json']
    trace_closed = read(TRACE/'closed.json')
    assert pin(TRACE/'closed.json')['sha256'] == '7778c67fa57d567ffa7d779b0960b78027f57fe1dff6bd448f7eca1c74542899'
    files = {}; worker_files = {}

    def bind(path, expected=None, worker=False):
        identity = pin(path)
        if expected is not None:
            assert identity == expected, str(path)
        files[rel(path)] = identity
        if worker:
            worker_files[rel(path)] = identity

    for path in [PRIOR/'manifest.json', PRIOR/'closed.json', TRACE/'closed.json']:
        bind(path)
    requests = copy.deepcopy(prior['requests'])
    assert [r['original'] for r in requests] == [0, 10, 9, 20]
    for request in requests:
        for desc in [request['input'], *request['baselines'].values(), *request['references']['numpy'], *request['references']['ort']]:
            bind(ROOT/desc['file'], prior['files'][desc['file']], worker=desc is request['input'])
        folder = TRACE/'outputs'/f"managed-{request['index']:02}-{request['name']}"
        result = read(folder/'result.json')
        bind(folder/'result.json', trace_closed['files'][rel(folder/'result.json').split(rel(TRACE)+'/')[1]])
        record = next(r for r in result['records'] if r['kind'] == 'MM')
        request['old_trace'] = []
        for row in record['outputs']:
            path = folder/row['file']; bind(path, trace_closed['files'][path.relative_to(TRACE).as_posix()])
            assert pin(path)['sha256'] == row['sha256']
            request['old_trace'].append(dict(file=rel(path), shape=row['shape'], name=row['name']))
    baseline = ROOT/prior['model']
    bind(baseline, prior['files'][rel(baseline)], worker=True)
    assert pin(baseline)['sha256'] == '0f45e6c282ead0d447f313727318714d2e5ec5910858bbad6de93c75cc483a96'
    model = onnx.load(baseline, load_external_data=False)
    assert len(model.graph.node) == 1559 and len(model.graph.output) == 41
    original = baseline.with_name('encoder_model.onnx')
    bind(original, prior['files'][rel(original)])
    untraced = onnx.load(original, load_external_data=False)
    del untraced.graph.output[:]; untraced.graph.output.extend(model.graph.output)
    assert untraced.SerializeToString(deterministic=True) == model.SerializeToString(deterministic=True)
    data = baseline.with_name('encoder_model.onnx_data')
    bind(data, prior['files'][rel(data)], worker=True)
    variant = append_erf_outputs(model)
    candidate = baseline.with_name('encoder_gelu_observed_20260921.onnx')
    assert not candidate.exists()
    with candidate.open('xb') as stream:
        stream.write(variant.SerializeToString(deterministic=True))
    bind(candidate, worker=True)
    models = {mode:dict(file=rel(path), outputs=output_specs(graph)) for mode,path,graph in
              [('baseline', baseline, model), ('unfused', candidate, variant)]}
    assert models['unfused']['outputs'][:41] == models['baseline']['outputs'] == prior['outputs']
    assert [o['shape'] for o in models['unfused']['outputs'][41:]] == [[1,1280,3000], [1,1280,1500]]+[[1,1500,5120]]*32
    assert pin(BASE/'bin/Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(BASE/'bin/Lokad.Onnx.dll') == pin(PRODUCT/'Lokad.Onnx.dll')
    for path in (BASE/'bin').iterdir():
        if path.is_file():
            bind(path, worker=True)
    for path in Path(__file__).parent.iterdir():
        if path.is_file():
            bind(path)
    bind(ROOT/'tests/Shared/NpySupport.cs')
    bind(BASE/'build.log')
    jobs = [dict(id=f'{i:02}-{mode}', request=i, mode=mode) for i in range(4) for mode in models]
    # Freeze analysis dependencies as well as inference binaries.
    numerical_files = {}
    for package in [Path(np.__file__).parent, Path(onnx.__file__).parent, Path(psutil.__file__).parent]:
        for path in package.rglob('*'):
            if path.is_file() and path.suffix in ['.py', '.pyd', '.dll']:
                numerical_files[str(path)] = pin(path)
    write(BASE/'manifest.json', dict(protocol=PROTOCOL, source_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        files=files, worker_files=worker_files, numerical_files=numerical_files, models=models, requests=requests, jobs=jobs,
        limits=LIMITS, core_sha256=CORE, calls=8, arrays=464, scaled_error_limit=1e-4, useful_maximum_ratio=.5))
    print(json.dumps(dict(prepared=True, files=len(files), manifest=pin(BASE/'manifest.json'))))


if __name__ == '__main__':
    assert sys.argv[1:] == ['prepare']
    prepare()
