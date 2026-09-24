"""Reconcile complete relative-position Slice/Reshape pairs; no inference."""
from collections import Counter
import csv
import hashlib
import json
import subprocess

import onnx
from onnx import numpy_helper

from run import ROOT, BASE, pin, read, write


def constants(graph):
    result = {tensor.name: tensor for tensor in graph.initializer}
    for node in graph.node:
        if node.op_type == 'Constant':
            value, = [a.t for a in node.attribute if a.name == 'value']
            output, = node.output
            result[output] = value
    return result


def main():
    closure = read(BASE / 'closed.json')
    assert closure['passed'] and closure['analysis'] == pin(BASE / 'analysis.json')
    managed = read(BASE / 'analysis.json')
    native_base = ROOT / 'artifacts/parakeet-ort-diagnosis-amd-20260924'
    assert read(native_base / 'closed.json')['analysis'] == pin(native_base / 'analysis.json')
    native = read(native_base / 'analysis.json')['profiles']['encoder']
    review_base = ROOT / 'artifacts/parakeet-ort-graph-review-20260924'
    assert read(review_base / 'closed.json')['analysis'] == pin(review_base / 'analysis.json')
    review = read(review_base / 'analysis.json')
    optimized_path = ROOT / 'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder/optimized.onnx'
    assert pin(optimized_path) == review['graphs']['encoder']['model']
    manifest = read(ROOT / 'artifacts/parakeet-prepared-recurrence-app-amd-20260924/collected/manifests/current-parakeet.json')
    original_path = ROOT / 'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    assert pin(original_path) == {k: manifest['models']['encoder-model.onnx'][k] for k in ['bytes', 'sha256']}
    original = onnx.load(original_path, load_external_data=False).graph
    optimized = onnx.load(optimized_path, load_external_data=False).graph
    original_nodes = {n.name: n for n in original.node}
    optimized_nodes = {n.name: n for n in optimized.node}
    original_constants, optimized_constants = constants(original), constants(optimized)
    managed_nodes = {n['name']: n for n in managed['phases']['wall']['node_rows'] if n['graph'] == 'encoder'}
    native_clocks = {n['name']: n for n in native['node_clocks']}
    frames = Counter(case['expected']['encoded_frames'] for case in manifest['cases'])
    expected_constants = [[1], [9223372036854775807], [2], [1]]
    rows = []
    for layer in range(24):
        prefix = f'/layers.{layer}/self_attn/'
        names = [prefix + 'Slice_1', prefix + 'Reshape_7']
        for name, op in zip(names, ['Slice', 'Reshape']):
            old, new = original_nodes[name], optimized_nodes[name]
            observed = managed_nodes[name]
            assert old.op_type == new.op_type == observed['op'] == native_clocks[name]['op'] == op
            assert list(old.output) == list(new.output) == observed['outputs']
            assert old.input[0] == new.input[0] == observed['inputs'][0]
            assert observed['calls'] == native_clocks[name]['calls'] == 60
            if op == 'Slice':
                for node, values in [(old, original_constants), (new, optimized_constants)]:
                    assert [numpy_helper.to_array(values[edge]).tolist() for edge in node.input[1:]] == expected_constants
                # Lokad also merges equal Constant nodes; compare their values.
                assert [numpy_helper.to_array(original_constants[edge]).tolist()
                        for edge in observed['inputs'][1:]] == expected_constants
            else:
                assert list(old.input) == list(new.input) == observed['inputs']
                assert old.input[0] == original_nodes[names[0]].output[0]
                assert all(a.i == 0 for a in new.attribute if a.name == 'allowzero')
            seen = Counter()
            for shape in native['shapes']:
                if shape['name'] != name:
                    continue
                source, = shape['inputs'][0].values()
                destination, = shape['outputs'][0].values()
                if op == 'Slice':
                    t = source[3]
                    assert source == [1, 8, 2*t, t] and destination == [1, 8, 2*t-1, t]
                else:
                    t = source[3]
                    assert source == [1, 8, 2*t-1, t] and destination == [1, 8, t, 2*t-1]
                seen[t] += shape['calls']
            # Native shape census includes warmup; clocks above exclude it.
            assert seen == Counter({t: count*4 for t, count in frames.items()})
        measured = sum(managed_nodes[name]['corpus_seconds'] for name in names)
        reference = sum(native_clocks[name]['exclusive_us'] for name in names) / 3e6
        rows.append(dict(layer=layer, managed_seconds=measured, ort_seconds=reference,
                         excess_seconds=measured-reference,
                         managed_slice_seconds=managed_nodes[names[0]]['corpus_seconds'],
                         managed_reshape_seconds=managed_nodes[names[1]]['corpus_seconds'],
                         ort_slice_seconds=native_clocks[names[0]]['exclusive_us']/3e6,
                         ort_reshape_seconds=native_clocks[names[1]]['exclusive_us']/3e6))
    revision = review['ort_revision']
    native_sources = {}
    for path in ['onnxruntime/core/providers/cpu/tensor/slice.cc',
                 'onnxruntime/core/providers/cpu/tensor/reshape.cc',
                 'onnxruntime/core/providers/cpu/tensor/reshape.h']:
        data = subprocess.check_output(['git', '-c', 'gc.auto=0', '-C', str(ROOT/'external/onnxruntime'),
                                        'show', revision+':'+path])
        native_sources[path] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    sources = {path: pin(ROOT/path) for path in ['src/Lokad.Onnx/TensorSlice.cs',
        'src/Lokad.Onnx/Tensor.cs', 'src/Lokad.Onnx/TensorOps.Shape.cs',
        'src/Lokad.Onnx/DenseTensor.cs', 'src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs']}
    summary = dict(passed=True, managed_closure=pin(BASE/'closed.json'),
        native_closure=pin(native_base/'closed.json'), graph_review=pin(review_base/'closed.json'),
        original_model=pin(original_path), layers=24, nodes_per_engine=48,
        measured_calls_per_node=60, frames=dict(sorted(frames.items())),
        managed_seconds=sum(r['managed_seconds'] for r in rows),
        ort_seconds=sum(r['ort_seconds'] for r in rows),
        excess_seconds=sum(r['excess_seconds'] for r in rows),
        managed_sources=sources, ort_revision=revision, native_sources=native_sources,
        runtime_managed_layout_observed=False, rows=rows)
    write(BASE/'slice-comparison.json', summary)
    out = ROOT/'tests/parakeet/managed-phase-results'
    with (out/'slices-20260924.csv').open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    compact = {k:v for k,v in summary.items() if k != 'rows'}
    with (out/'slice-observations-20260924.json').open('x', encoding='utf8') as stream:
        json.dump(compact, stream, indent=2); stream.write('\n')
    print(json.dumps(compact))


if __name__ == '__main__':
    main()
