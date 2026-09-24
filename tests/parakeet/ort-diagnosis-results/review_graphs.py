"""Join executed ORT nodes, serialized original-output graphs and matching source."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import onnx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/ort-diagnosis-amd'))
from run import BASE as PROFILE, pin, read, write


def main():
    out = ROOT/'artifacts/parakeet-ort-graph-review-20260924'
    assert not out.exists(); out.mkdir()
    profile = read(PROFILE/'analysis.json')
    assert pin(PROFILE/'analysis.json') == read(PROFILE/'closed.json')['analysis']
    encoder = ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924'
    small = ROOT/'artifacts/parakeet-ort-small-graphs-amd-20260924'
    assert not read(encoder/'closed.json')['passed'] and read(small/'closed.json')['passed']
    reports, resources, projections = {}, {}, []
    for graph, base in [('encoder',encoder),('decoder',small),('frontend',small)]:
        transfer = read(base/'transfer.json')
        for name, wanted in transfer['files'].items():
            assert pin(base/'collected'/name) == wanted
        row = next(r for r in transfer['state']['runs'] if r['graph'] == graph)
        spec = read(base/'prepared.json'); limits = spec['limits']
        assert row['complete'] and row['code'] == 0 and row['seconds'] < limits['seconds']
        assert row['preflight']['available'] >= limits['preflight_available'] and row['preflight']['tmpfs'] >= limits['preflight_tmpfs']
        samples = row['samples']; assert samples
        for sample in samples:
            assert sample['rss'] < limits['rss'] and sample['available'] >= limits['available']
            assert sample['tmpfs'] >= limits['tmpfs'] and sample['output'] < limits['output']
            assert sample['seconds'] < limits['seconds'] and all(a == [2] for a in sample['threads'])
        gaps = [samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0 <= v < 10 for v in gaps)
        folder = base/'collected'/graph; result = read(folder/'result.json')
        assert result['scratch_retired'] and result['inference_calls'] == 0
        model_path = folder/'optimized.onnx'; assert pin(model_path) == result['optimized_model']
        model = onnx.load(model_path, load_external_data=False)
        nodes = {n.name:n for n in model.graph.node}; initializers = {t.name:t for t in model.graph.initializer}
        observed = profile['profiles'][graph]
        assert len(nodes) == len(model.graph.node) and set(nodes) == set(observed['nodes'])
        assert all(n.op_type == observed['nodes'][name]['op'] for name,n in nodes.items())
        setup = next(r for r in read(PROFILE/'collected/control/observation.json')['setup'] if r['graph'] == graph)
        assert result['inputs'] == setup['inputs'] and result['outputs'] == setup['outputs']
        for name,node in nodes.items():
            if graph != 'encoder' or node.op_type not in ('MatMul','FusedMatMul'):
                continue
            assert len(node.input) == 2
            constant = node.input[1] in initializers
            shape = list(initializers[node.input[1]].dims) if constant else None
            if constant:
                assert initializers[node.input[1]].data_type == onnx.TensorProto.FLOAT and len(shape) == 2
            shapes = [r for r in observed['shapes'] if r['name'] == name]
            assert shapes and all(len(r['inputs']) == (1 if constant else 2) for r in shapes)
            time_row = next(r for r in observed['node_clocks'] if r['name'] == name)
            attrs = {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            projections.append(dict(name=name,op=node.op_type,constant_b=constant,
                b_name=node.input[1],k=None if shape is None else shape[0],n=None if shape is None else shape[1],
                source_bytes=0 if shape is None else shape[0]*shape[1]*4,
                alpha=attrs.get('alpha',1.0),runtime_input_counts=sorted({len(r['inputs']) for r in shapes}),
                corpus_seconds=time_row['exclusive_us']/3e6,shape_observations=len(shapes)))
        reports[graph] = dict(nodes=len(nodes),census=dict(Counter(n.op_type for n in nodes.values())),
            model=result['optimized_model'],retired_scratch=result['retired_scratch'],executed_nodes_exact=True,
            io_descriptors_exact=True,artifact=str(base.relative_to(ROOT)))
        resources[graph] = dict(samples=len(samples),peak_rss=max(s['rss'] for s in samples),seconds=row['seconds'],owner=row['owner'])
    revision = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
    sources = {}
    for name in ['onnxruntime/core/framework/sequential_executor.cc','onnxruntime/core/framework/session_state.cc',
                 'onnxruntime/core/providers/cpu/math/matmul.cc','onnxruntime/contrib_ops/cpu/fused_matmul.cc']:
        command = ['git','-C',str(ROOT/'external/onnxruntime'),'show',revision+':'+name]
        result = subprocess.run(command,check=True,capture_output=True)
        path = out/Path(name).name; path.write_bytes(result.stdout)
        sources[name] = pin(path)
    source_text = {p.name:p.read_text() for p in out.glob('*.cc')}
    assert 'p_input != nullptr && p_input->IsAllocated() && p_input->IsTensor()' in source_text['sequential_executor.cc']
    assert 'st->initialized_tensors_.erase(ort_value_idx)' in source_text['session_state.cc']
    assert 'data[i].BIsPacked = bool(packed_b_)' in source_text['matmul.cc']
    assert 'MatMul<float>);' in source_text['fused_matmul.cc']
    constant = [p for p in projections if p['constant_b']]
    dynamic = [p for p in projections if not p['constant_b']]
    assert len(constant) == 217 and len(dynamic) == 72
    assert sum(p['op'] == 'FusedMatMul' for p in constant) == 48
    assert all(p['alpha'] == .5 for p in constant if p['op'] == 'FusedMatMul')
    analysis = dict(passed=True,graphs=reports,resources=resources,ort_revision=revision,sources=sources,
        projections=projections,constant_projections=len(constant),dynamic_projections=len(dynamic),
        constant_projection_seconds=sum(p['corpus_seconds'] for p in constant),
        dynamic_projection_seconds=sum(p['corpus_seconds'] for p in dynamic),
        constant_source_bytes=sum(p['source_bytes'] for p in constant),
        preparation_inference='Nonempty FP32 MatMul/FusedMatMul B is absent from all runtime input profiles; matching source releases prepacked initializers and consumes packed_b_. This supports prepared-B use; it does not identify the dispatched MLAS leaf.',
        original_encoder_stage_refusal=pin(encoder/'closed.json'),small_graph_closure=pin(small/'closed.json'),
        native_profile_closure=pin(PROFILE/'closed.json'),inference_calls=0)
    write(out/'analysis.json',analysis)
    write(out/'closed.json',dict(passed=True,analysis=pin(out/'analysis.json'),files={p.name:pin(p) for p in out.iterdir() if p.is_file()}))
    report = Path(__file__).resolve().parent
    with (report/'ort-projections-20260924.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(projections[0]),lineterminator='\n');writer.writeheader();writer.writerows(projections)
    with (report/'ort-graphs-20260924.json').open('x',encoding='utf8') as stream:
        json.dump(dict(closure=pin(out/'closed.json'),**{k:v for k,v in analysis.items() if k!='projections'}),stream,indent=2)
    print(json.dumps({k:analysis[k] for k in ['passed','constant_projections','dynamic_projections','constant_projection_seconds','dynamic_projection_seconds','constant_source_bytes']}))


if __name__ == '__main__':
    main()
