"""Count Conv output payload from pinned metadata, without running inference."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/pyannote-convolution-allocation-20260921'
PROBE = ROOT / 'artifacts/pyannote-context-reuse-probe-20260921'
PROFILE = ROOT / 'artifacts/pyannote-performance-profile-20260921'
CORE = ROOT / 'artifacts/pyannote-lstm-output-lanes-20260921'
sys.path.insert(0, str(ROOT / 'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil
import onnx


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    assert not BASE.exists()
    own = psutil.Process()
    previous_affinity = own.cpu_affinity()
    own.cpu_affinity([0])
    try:
        closures = [
            (PROBE / 'closed.json', 'd06bffe6d4aba50e7b17373c2a155234c514419bd4859063938fb1fe436fb626'),
            (PROFILE / 'closed.json', '4b7542a8917e558c92ba632fada6830ddf8c870fccf49064dae17ff1de194d5c'),
            (CORE / 'qualification-closed.json', 'cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53')]
        trusted, files = {}, {}
        for path, sha in closures:
            assert pin(path)['sha256'] == sha
            closed = read(path)
            assert closed['passed']
            trusted.update({str((ROOT / name).resolve()): wanted for name, wanted in closed['files'].items()})
            files[str(path.relative_to(ROOT))] = pin(path)

        def checked(path):
            identity = pin(path)
            assert identity == trusted[str(path.resolve())], path
            files[str(path.relative_to(ROOT))] = identity
            return path

        manifest = read(checked(PROBE / 'manifest.json'))
        profile = read(checked(PROFILE / 'output/result.json'))
        allocation = read(checked(PROBE / 'analysis.json'))
        model_path = checked(ROOT / manifest['models']['embedding']['path'])
        assert pin(model_path) == {k: manifest['models']['embedding'][k] for k in ['bytes', 'sha256']}
        for name in ['CPUExecutionProvider.ConvPool.cs', 'CPUExecutionProvider.Fusion.cs', 'TensorOps.ConvPool.cs',
                     'DenseTensor.cs', 'ComputationalGraph.cs', 'Node.cs', 'TensorBufferPool.cs']:
            checked(CORE / 'candidate-source/src/Lokad.Onnx' / name)
        cases = [c for c in manifest['cases'] if c['graph'] == 'embedding']
        assert len(cases) == 3
        results = []
        for case in cases:
            checked(ROOT / case['input']['path'])
            checked(ROOT / case['expected']['path'])
            model = onnx.load(model_path, load_external_data=False)
            assert len(model.graph.input) == 1 and model.graph.input[0].name == case['input_name']
            del model.graph.value_info[:]
            for output in model.graph.output:
                output.type.tensor_type.ClearField('shape')
            for dim, size in zip(model.graph.input[0].type.tensor_type.shape.dim, case['input']['shape'], strict=True):
                dim.ClearField('dim_param')
                dim.dim_value = size
            model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
            values = {v.name: v for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}

            def shape(name):
                dims = values[name].type.tensor_type.shape.dim
                assert all(d.HasField('dim_value') and d.dim_value > 0 for d in dims), name
                return [d.dim_value for d in dims]

            weights = {v.name: list(v.dims) for v in model.graph.initializer}
            captured = [r for r in profile['rows'] if r['name'] == case['name'] and r['model'] == 'embedding' and r['phase'] == 'wall']
            assert len(captured) == 1 and captured[0]['input']['shape'] == case['input']['shape']
            executed = {n['name']: n['op'] for n in captured[0]['nodes'] if n['op'] in ['Conv', 'ConvRelu']}
            rows = []
            for node in model.graph.node:
                if node.op_type != 'Conv':
                    continue
                x, w, y = shape(node.input[0]), weights[node.input[1]], shape(node.output[0])
                assert len(x) == len(w) == len(y) == 4 and values[node.output[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                assert attrs.get('auto_pad', b'NOTSET') in [b'NOTSET', b'']
                pads, strides, dilations = attrs.get('pads', [0, 0, 0, 0]), attrs.get('strides', [1, 1]), attrs.get('dilations', [1, 1])
                assert x[1] == w[1] * attrs.get('group', 1)
                manual = [x[0], w[0]] + [(x[i+2] + pads[i] + pads[i+2] - (dilations[i] * (w[i+2]-1) + 1)) // strides[i] + 1 for i in range(2)]
                assert manual == y and node.name in executed
                rows.append(dict(name=node.name, executed_op=executed[node.name], input_shape=x, weight_shape=w,
                    output_shape=y, output=node.output[0], payload_bytes=math.prod(y) * 4, attributes=attrs))
            assert len(rows) == 36 and {r['name'] for r in rows} == set(executed)
            assert shape(case['output_name']) == case['expected']['shape']
            total = sum(r['payload_bytes'] for r in rows)
            groups = collections.Counter((tuple(r['output_shape']), r['payload_bytes']) for r in rows)
            results.append(dict(name=case['name'], input_shape=case['input']['shape'], convolution_payload_bytes=total, rows=rows,
                groups=[dict(shape=list(s), each_payload_bytes=b, count=n, total_payload_bytes=b*n) for (s, b), n in groups.items()]))
        assert len({r['convolution_payload_bytes'] for r in results}) == 1
        references = []
        for order in ['forward', 'reverse']:
            rows = [r for r in allocation['rows'] if r['order'] == order and r['graph'] == 'embedding' and r['mode'] == 'reuse' and r['phase'] == 'repeat']
            assert len(rows) == 6
            mean = statistics.fmean(r['allocated_bytes'] for r in rows)
            references.append(dict(order=order, calls=6, measured_allocation_mean=mean,
                static_conv_payload_fraction=results[0]['convolution_payload_bytes'] / mean))
        for path in [Path(__file__), Path(onnx.__file__), Path(onnx.shape_inference.__file__)]:
            files[str(path)] = pin(path)
        analysis = dict(passed=True, inference_executed=False, onnx=onnx.__version__, cases=results, retained_allocation=references,
            scope='Static output payload with independent Conv geometry and retained executed-node coverage; no new allocation measurement or speed claim.')
        BASE.mkdir()
        target = BASE / 'analysis.json'
        target.write_text(json.dumps(analysis, indent=2) + '\n', encoding='utf8')
        files[str(target.relative_to(ROOT))] = pin(target)
        (BASE / 'closed.json').write_text(json.dumps(dict(passed=True, files=files, analysis=pin(target)), indent=2) + '\n', encoding='utf8')
        print(json.dumps(dict(passed=True, cases=len(results), nodes_each=36, payload_bytes=results[0]['convolution_payload_bytes'],
            retained_allocation=references, closed=pin(BASE / 'closed.json'))))
    finally:
        own.cpu_affinity(previous_affinity)


if __name__ == '__main__':
    main()
