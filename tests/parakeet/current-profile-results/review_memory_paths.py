"""Inspect retained graph constants and pinned sources without running inference."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts/parakeet-memory-source-review-20260923'
ORT = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def inspect_graph(path):
    # In particular, do not load/copy the 2.3 GB external encoder weights.
    model = onnx.load(path, load_external_data=False)
    producers = {output: node for node in model.graph.node for output in node.output}
    memo = {}

    def constant(name):
        if name in memo:
            return memo[name]
        node = producers[name]
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        args = [constant(i) for i in node.input]
        if node.op_type == 'Constant':
            result = numpy_helper.to_array(attrs['value'])
        elif node.op_type == 'ConstantOfShape':
            value = numpy_helper.to_array(attrs['value'])
            assert value.size == 1 and np.prod(args[0]) <= 64
            result = np.full(tuple(args[0]), value.item(), dtype=value.dtype)
        elif node.op_type == 'Concat':
            result = np.concatenate(args, axis=attrs['axis'])
        elif node.op_type == 'Reshape':
            assert not attrs.get('allowzero', 0) and np.all(args[1] != 0)
            result = args[0].reshape(tuple(args[1]))
        elif node.op_type == 'Slice':
            slices = [slice(None)] * args[0].ndim
            for start, end, axis, step in zip(*args[1:], strict=True):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            result = args[0][tuple(slices)]
        elif node.op_type == 'Transpose':
            result = args[0].transpose(attrs['perm'])
        elif node.op_type == 'Cast':
            assert attrs['to'] == onnx.TensorProto.INT64
            result = args[0].astype(np.int64)
        else:
            raise AssertionError(('Not a permitted constant expression', node.name, node.op_type))
        assert result.size <= 64
        memo[name] = result
        return result

    pads = []
    wheres = []
    for node in model.graph.node:
        if node.op_type == 'Pad':
            mode = next(a.s.decode() for a in node.attribute if a.name == 'mode')
            values = constant(node.input[1]).tolist()
            rank = len(values) // 2
            assert mode == 'constant' and len(values) == rank * 2
            assert len(node.input) == 3 and node.input[2] == ''
            assert all(values[d] == values[d + rank] == 0 for d in range(rank - 1))
            assert values[rank - 1] >= 0 and values[-1] >= 0
            pads.append(dict(name=node.name, data=node.input[0], pads=values,
                             mode=mode, fill='omitted (zero)', node_sha256=hashlib.sha256(node.SerializeToString()).hexdigest()))
        elif node.op_type == 'Where':
            scalar = None
            branch = producers.get(node.input[1])
            if branch is not None and branch.op_type == 'Constant':
                value = numpy_helper.to_array(next(a.t for a in branch.attribute if a.name == 'value'))
                if value.ndim == 0:
                    scalar = dict(dtype=str(value.dtype), value=value.item())
            wheres.append(dict(name=node.name, inputs=list(node.input), x_scalar=scalar))
    assert len(pads) == 48 and len(wheres) == 73
    counts = collections.Counter(tuple(row['pads']) for row in pads)
    assert counts == {(0, 0, 0, 1, 0, 0, 0, 0): 24, (0, 0, 4, 0, 0, 4): 24}
    return dict(model=pin(path), nodes=len(model.graph.node), pads=pads, wheres=wheres,
                constant_expression_values=len(memo), external_weights_loaded=False,
                inference_executed=False)


def main():
    assert not ART.exists()
    target = OUT / 'memory-source-observations-20260923.json'
    assert not target.exists()
    ort_files = [
        'onnxruntime/core/providers/cpu/tensor/pad.cc',
        'onnxruntime/core/providers/cpu/tensor/where_op.cc',
        'onnxruntime/core/mlas/lib/mlasi.h',
        'onnxruntime/core/mlas/lib/sgemm.cpp',
    ]
    local_files = [
        'src/Lokad.Onnx/CPUExecutionProvider.Shape.cs',
        'src/Lokad.Onnx/TensorOps.Elementwise.cs',
        'src/Lokad.Onnx/Tensor.cs',
        'src/Lokad.Onnx/TensorSlice.cs',
        'src/Lokad.Onnx/BroadcastedTensor.cs',
        'src/Lokad.Onnx/MathOps.cs',
        'tests/parakeet/wide-matmul/Blocked.cs',
    ]
    graph = inspect_graph(ROOT / 'models/parakeet-tdt-0.6b-v3/encoder-model.onnx')
    sources = {}
    for name in ort_files:
        sources['ort/' + name] = subprocess.run(
            ['git', '-C', str(ROOT / 'external/onnxruntime'), 'show', ORT + ':' + name],
            capture_output=True, check=True).stdout
    for name in local_files:
        sources['lokad/' + name] = (ROOT / name).read_bytes()
    ART.mkdir()
    for name, data in sources.items():
        path = ART / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    result = dict(ort_commit=ORT, product='94a550de', graph=graph,
                  sources={name: pin(ART / name) for name in sources},
                  profile=pin(OUT / 'observations-20260923.json'),
                  generator=pin(Path(__file__)),
                  scope='Static graph/constants and source inspection; no new timings or native dispatch claim.')
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(pads=len(graph['pads']), wheres=len(graph['wheres']),
                         scalar_x_wheres=sum(r['x_scalar'] is not None for r in graph['wheres']),
                         sources=len(sources), observations=pin(target))))


if __name__ == '__main__':
    main()
