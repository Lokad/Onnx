"""Match masking/padding and their folded inputs from retained graphs; no inference."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import numpy_helper

from run import ROOT, BASE, pin, read, write
from compare_slices import constants


def attributes(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class Graph:
    def __init__(self, graph, observed=None, clocks=None):
        self.proto = {n.name: n for n in graph.node}
        self.constants = constants(graph)
        self.initializers = {t.name for t in graph.initializer}
        self.nodes = observed if observed is not None else {
            n.name: dict(name=n.name, op=n.op_type, inputs=list(n.input), outputs=list(n.output),
                calls=clocks[n.name]['calls'], corpus_seconds=clocks[n.name]['exclusive_us']/3e6)
            for n in graph.node}
        self.producers = {edge: n for n in self.nodes.values() for edge in n['outputs'] if edge}
        assert len(self.producers) == sum(bool(e) for n in self.nodes.values() for e in n['outputs'])
        self.cache = {}

    def attrs(self, node):
        original = self.proto[node['name']]
        assert node['op'] == original.op_type and node['outputs'] == list(original.output)
        return attributes(original)

    def constant(self, edge):
        """Evaluate only the model's small integer metadata, never model arithmetic."""
        if edge in self.cache: return self.cache[edge]
        if edge in self.constants:
            tensor = self.constants[edge]
            assert np.prod(tensor.dims, dtype=np.int64) <= 64 and not tensor.external_data
            result = numpy_helper.to_array(tensor)
        else:
            node = self.producers[edge]; op = node['op']; attrs = self.attrs(node)
            assert op in ['Cast', 'Reshape', 'Transpose', 'Slice', 'Concat', 'ConstantOfShape'], (edge, op)
            args = [self.constant(e) for e in node['inputs'] if e]
            assert all(a.dtype == np.dtype('int64') and a.size <= 64 for a in args)
            if op == 'Cast':
                assert attrs == {'to': onnx.TensorProto.INT64}; result = args[0]
            elif op == 'Reshape':
                assert attrs.get('allowzero', 0) == 0 and np.all(args[1] != 0)
                result = args[0].reshape(args[1].tolist())
            elif op == 'Transpose': result = args[0].transpose(attrs['perm'])
            elif op == 'Concat': result = np.concatenate(args, axis=attrs['axis'])
            elif op == 'ConstantOfShape':
                assert np.prod(args[0], dtype=np.int64) <= 64
                fill = numpy_helper.to_array(attrs['value']); assert fill.dtype == np.dtype('int64') and fill.size == 1
                result = np.full(args[0].tolist(), fill.item(), dtype=np.int64)
            else:
                assert len(args) == 5
                slices = [slice(None)]*args[0].ndim
                for start, end, axis, step in zip(*[a.tolist() for a in args[1:]], strict=True):
                    assert step in [-1, 1]
                    # ONNX negative-step INT64_MIN means before the first element.
                    slices[axis] = slice(start, None if step < 0 and end == -(2**63) else end, step)
                result = args[0][tuple(slices)]
        assert result.size <= 64
        self.cache[edge] = result
        return result

    def condition(self, edge):
        node = self.producers[edge]
        if node['op'] == 'Cast':
            assert self.attrs(node) == {'to': onnx.TensorProto.BOOL}
            node = self.producers[node['inputs'][0]]
        assert node['op'] == 'Unsqueeze' and not self.attrs(node)
        assert self.constant(node['inputs'][1]).tolist() == [1]
        boundary = node['inputs'][0]
        assert boundary in ['/Not_output_0', '/Not_1_output_0']
        producer = self.producers[boundary]
        assert producer['op'] == 'Not' and not self.attrs(producer)
        return boundary

    def ancestors(self, output, boundary):
        found = {}; leaves = set(); visiting = set()
        def visit(edge):
            if not edge: return
            if edge in boundary or edge in self.initializers:
                leaves.add(edge); return
            node = self.producers[edge]; name = node['name']
            assert name not in visiting
            if name in found: return
            assert node['calls'] == 60
            visiting.add(name)
            for source in node['inputs']: visit(source)
            visiting.remove(name); found[name] = node
        visit(output)
        assert leaves & boundary == boundary
        return found


def closed(folder):
    proof = read(folder/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(folder/'analysis.json')
    return read(folder/'analysis.json')


def summarize(nodes):
    operators = defaultdict(lambda: dict(nodes=0, seconds=0.))
    for n in nodes.values():
        operators[n['op']]['nodes'] += 1
        operators[n['op']]['seconds'] += n['corpus_seconds']
    return dict(nodes=len(nodes), seconds=sum(n['corpus_seconds'] for n in nodes.values()), operators=dict(operators))


def main():
    managed = closed(BASE)
    native_base = ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924'; native = closed(native_base)
    review_base = ROOT/'artifacts/parakeet-ort-graph-review-20260924'; review = closed(review_base)
    optimized_path = ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder/optimized.onnx'
    original_path = ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    manifest = read(ROOT/'artifacts/parakeet-prepared-recurrence-app-amd-20260924/collected/manifests/current-parakeet.json')
    assert pin(optimized_path) == review['graphs']['encoder']['model']
    assert pin(original_path) == {k: manifest['models']['encoder-model.onnx'][k] for k in ['bytes', 'sha256']}
    original = onnx.load(original_path, load_external_data=False).graph
    optimized = onnx.load(optimized_path, load_external_data=False).graph
    observations = native['profiles']['encoder']
    clocks = {n['name']: n for n in observations['node_clocks']}
    assert set(clocks) == {n.name for n in optimized.node}
    observed = {n['name']: n for n in managed['phases']['wall']['node_rows'] if n['graph'] == 'encoder'}
    graphs = dict(managed=Graph(original, observed), ort=Graph(optimized, clocks=clocks))
    frames = Counter(c['expected']['encoded_frames'] for c in manifest['cases'])
    assert len(frames) == 19 and sum(frames.values()) == 20
    rows = []; unions = {label: {} for label in graphs}; families = {}; memberships = defaultdict(list)
    kinds = [('attention-mask', 'self_attn/Where'), ('attention-cleanup', 'self_attn/Where_1'),
             ('convolution-mask', 'conv/Where'), ('attention-pad', 'self_attn/Pad'), ('convolution-pad', 'conv/depthwise_conv/Pad')]
    for kind, suffix in kinds:
        family = {label: {} for label in graphs}
        for layer in range(24):
            name = f'/layers.{layer}/{suffix}'; is_where = 'pad' not in kind; attention = kind.startswith('attention')
            expected_op = 'Where' if is_where else 'Pad'
            nodes = {label: g.nodes[name] for label, g in graphs.items()}
            output = nodes['managed']['outputs'][0]
            assert all(n['op'] == expected_op and n['outputs'] == [output] and n['calls'] == 60 for n in nodes.values())
            boundary = set(); true_bits = None; pad_values = None
            for label, g in graphs.items():
                node = nodes[label]; inputs = node['inputs']; attrs = g.attrs(node)
                if is_where:
                    assert len(inputs) == 3 and not attrs
                    mask = g.condition(inputs[0]); assert mask == ('/Not_output_0' if attention else '/Not_1_output_0')
                    value = g.constant(inputs[1]); assert value.shape == () and value.dtype == np.dtype('float32')
                    wanted = np.array(-10000. if kind == 'attention-mask' else 0., dtype=np.float32)
                    assert value.tobytes() == wanted.tobytes(); true_bits = value.tobytes().hex()
                    cuts = {mask, inputs[2]}
                else:
                    assert len(inputs) == 3 and not inputs[2] and attrs == {'mode': b'constant'}
                    value = g.constant(inputs[1]); assert value.dtype == np.dtype('int64')
                    wanted = [0,0,0,1,0,0,0,0] if attention else [0,0,4,0,0,4]
                    assert value.tolist() == wanted, (label, name, value.tolist())
                    pad_values = wanted; cuts = {inputs[0]}
                if not boundary: boundary = cuts
                assert boundary == cuts
                found = g.ancestors(output, boundary)
                family[label].update(found); unions[label].update(found)
                for key in found: memberships[label, key].append(name)
            seen = Counter()
            for shape in observations['shapes']:
                if shape['name'] != name: continue
                ins = shape['inputs']; outs = shape['outputs']; t = outs[0]['float'][2] if attention else outs[0]['float'][2] - (8 if not is_where else 0)
                if is_where:
                    data = [1,8,t,t] if attention else [1,1024,t]
                    condition = [1,1,t,t] if attention else [1,1,t]
                    assert ins == [{'bool':condition},{'float':[]},{'float':data}] and outs == [{'float':data}]
                elif attention:
                    assert ins == [{'float':[1,8,t,2*t-1]},{'int64':[8]}] and outs == [{'float':[1,8,t,2*t]}]
                else:
                    assert ins == [{'float':[1,1024,t]},{'int64':[6]}] and outs == [{'float':[1,1024,t+8]}]
                seen[t] += shape['calls']
            assert seen == Counter({t: n*4 for t,n in frames.items()})
            measured = nodes['managed']['corpus_seconds']; reference = nodes['ort']['corpus_seconds']
            rows.append(dict(kind=kind,layer=layer,name=name,managed_seconds=measured,ort_seconds=reference,
                excess_seconds=measured-reference,boundary=sorted(boundary),output=output,true_scalar_bits=true_bits,pads=pad_values,measured_calls=60,shape_observations=len(seen)))
        families[kind] = {label: summarize(nodes) for label,nodes in family.items()}
    assert len(rows) == 120
    shared = {label: [dict(name=name,members=members,seconds=unions[label][name]['corpus_seconds'])
        for (engine,name),members in memberships.items() if engine == label and len(members)>1] for label in graphs}
    sources = {name: pin(ROOT/name) for name in ['src/Lokad.Onnx/CPUExecutionProvider.Shape.cs',
        'src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs','src/Lokad.Onnx/TensorOps.Elementwise.cs']}
    native_sources = {}; revision = review['ort_revision']
    for name in ['onnxruntime/core/providers/cpu/tensor/where_op.cc','onnxruntime/core/providers/cpu/tensor/pad.cc',
                 'onnxruntime/core/providers/cpu/tensor/utils.h']:
        data = subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),'show',revision+':'+name])
        native_sources[name] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    result = dict(passed=True,managed_closure=pin(BASE/'closed.json'),native_closure=pin(native_base/'closed.json'),
        graph_review=pin(review_base/'closed.json'),original_model=pin(original_path),optimized_model=pin(optimized_path),
        families=families,combined={label:summarize(nodes) for label,nodes in unions.items()},shared_nodes=shared,
        rows=rows,frames=dict(sorted(frames.items())),managed_sources=sources,ort_revision=revision,native_sources=native_sources,
        helpers={name:pin(Path(__file__).parent/name) for name in ['compare_slices.py','run.py']},
        new_inference_calls=0,metadata_only_constant_evaluation=True,new_optimization_selected=False,
        runtime_mask_values_captured=False,managed_runtime_layout_observed_for_this_corpus=False,
        shared_ancestors_counted_once=True,profiler_clocks_are_diagnostic=True,
        caveat='These selected-release profiles predate the composition. Family ancestor totals overlap at shared constants; use the combined union, not their sum. Existing convolution attribution also contains these convolution masking/padding nodes.')
    out = ROOT/'artifacts/parakeet-masking-padding-attribution-20260924'; assert not out.exists(); out.mkdir()
    write(out/'analysis.json', result)
    write(out/'closed.json', dict(passed=True,analysis=pin(out/'analysis.json'),auditor=pin(__file__),new_inference_calls=0))
    report = ROOT/'tests/parakeet/managed-phase-results'
    with (report/'masking-padding-20260924.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (report/'masking-padding-observations-20260924.json').open('x',encoding='utf8') as f:
        json.dump(dict(closure=pin(out/'closed.json'),**{k:v for k,v in result.items() if k!='rows'}),f,indent=2);f.write('\n')
    print(json.dumps(dict(closure=pin(out/'closed.json'),families=families,combined=result['combined'])))


if __name__ == '__main__': main()
