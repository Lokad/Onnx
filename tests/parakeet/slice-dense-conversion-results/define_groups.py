"""Define complete positional work from the exact graphs before candidate timing."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import onnx
from onnx import numpy_helper

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BOUNDARY={'/Gather_output_0','onnx::Slice_780'}
TARGETS=[f'/layers.{i}/self_attn/linear_pos/MatMul' for i in range(24)]


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf8'))


def scope(graph,observed=None):
    nodes=observed if observed is not None else {n.name:dict(name=n.name,op=n.op_type,inputs=list(n.input),outputs=list(n.output)) for n in graph.node}
    producers={e:n for n in nodes.values() for e in n['outputs'] if e}
    initializers={t.name for t in graph.initializer};groups={};union={};membership=Counter()
    for name in TARGETS:
        target=nodes[name];assert target['op']=='MatMul' and target['inputs'][0]=='/pos_enc/Slice_output_0'
        found={};leaves=set();visiting=set()
        def visit(edge):
            if not edge:return
            if edge in BOUNDARY:leaves.add(edge);return
            if edge in initializers:return
            node=producers[edge];key=node['name'];assert key not in visiting
            if key in found:return
            visiting.add(key)
            for source in node['inputs']:visit(source)
            visiting.remove(key);found[key]=node
        assert len(target['outputs'])==1;visit(target['outputs'][0]);assert leaves==BOUNDARY
        groups[name]=sorted(found);union.update(found);membership.update(found.keys())
    assert sum(n['op']=='MatMul' for n in union.values())==24
    assert {n['name'] for n in union.values() if n['op']=='Slice'}=={'/pos_enc/Slice'}
    shared={n for n,c in membership.items() if c==24};assert shared==set(union)-set(TARGETS)
    return dict(boundary=sorted(BOUNDARY),targets=TARGETS,per_projection=groups,
        shared_nodes=sorted(shared),shared_counted_once=True,
        nodes=[dict(**{k:n[k] for k in ['name','op','inputs','outputs']},members=membership[n['name']]) for n in sorted(union.values(),key=lambda n:n['name'])])


def main():
    output=OUT/'groups-20260925.json';assert not output.exists()
    proof=read(ROOT/'tests/parakeet/observed-dense-where-results/projection-copy-20260924.json');assert proof['passed']
    source=ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    native=ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder/optimized.onnx'
    assert pin(source)==proof['source_model'] and pin(native)==proof['native_optimized_model']
    models={label:onnx.load(path,load_external_data=False).graph for label,path in [('managed',source),('ort',native)]}
    managed=ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924'
    ort=ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924'
    for folder in [managed,ort]:
        closed=read(folder/'closed.json');assert closed['passed'] and closed['analysis']==pin(folder/'analysis.json')
    old_managed=read(managed/'analysis.json')['phases']['candidate']['node_rows']
    clocks=dict(managed={r['name']:r for r in old_managed if r['graph']=='encoder'},
        ort={r['name']:r for r in read(ort/'analysis.json')['profiles']['encoder']['node_clocks']})
    groups=dict(managed=scope(models['managed'],clocks['managed']),ort=scope(models['ort']))
    raw_scope=scope(models['managed'])
    assert len(raw_scope['nodes'])==37 and len(groups['managed']['nodes'])==34 and len(groups['ort']['nodes'])==30
    assert len(groups['managed']['shared_nodes'])==10 and len(groups['ort']['shared_nodes'])==6
    proto={n.name:n for n in models['managed'].node};constants={}
    for n in models['managed'].node:
        if n.op_type=='Constant':
            tensors=[a.t for a in n.attribute if a.HasField('t')]
            if len(tensors)==1:constants[n.output[0]]=tensors[0]
    aliases=[]
    for n in groups['managed']['nodes']:
        original=proto[n['name']];assert n['op']==original.op_type and n['outputs']==list(original.output)
        for before,after in zip(original.input,n['inputs'],strict=True):
            if before==after:continue
            values=[numpy_helper.to_array(constants[e]) for e in [before,after]]
            a,b=values;assert a.size<=64 and b.size<=64 and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
            aliases.append(dict(node=n['name'],original=before,executed=after,shape=list(a.shape),dtype=str(a.dtype),bits=a.tobytes().hex()))
    assert len(aliases)==6
    baseline={}
    for label,group in groups.items():
        rows=[]
        for node in group['nodes']:
            clock=clocks[label][node['name']];assert clock['calls']==60
            if label=='managed':
                assert all(node[k]==clock[k] for k in ['name','op','inputs','outputs'])
                seconds=clock['corpus_seconds']
            else:
                assert clock['inclusive_us']==clock['exclusive_us'];seconds=clock['exclusive_us']/3e6
            rows.append(dict(name=node['name'],seconds=seconds,projection=node['name'] in TARGETS))
        baseline[label]=dict(complete_seconds=sum(r['seconds'] for r in rows),projection_seconds=sum(r['seconds'] for r in rows if r['projection']),rows=rows)
    result=dict(passed=True,no_inference=True,source_model=pin(source),native_graph=pin(native),copy_proof=pin(ROOT/'tests/parakeet/observed-dense-where-results/projection-copy-20260924.json'),
        groups=groups,raw_export_nodes=len(raw_scope['nodes']),verified_constant_aliases=aliases,
        earlier_profile_context=baseline,earlier_profiles_are_not_candidate_measurements=True,
        earlier_closures=dict(managed=pin(managed/'closed.json'),ort=pin(ort/'closed.json')),
        prospective_prediction='The complete positional group and every one of its 24 MatMul kernels improve. Report all other node/phase clocks. No observer overhead subtraction or application score.',
        application_gate_unchanged=dict(minimum_corpus_gain=.03,maximum_clip_regression=.05,corpus_repeatability=1.10,clip_repeatability=1.20),
        attribution_boundary='From the already computed encoded-frame scalar and positional constant through shared slice geometry/materialization and all24 projection outputs. Weight initializers are boundaries; runtime input copies and packing inside MatMul remain in its interval.',
        reviewer=pin(Path(__file__)))
    with output.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,scope=pin(output),nodes={k:len(v['nodes']) for k,v in groups.items()},earlier_seconds={k:v['complete_seconds'] for k,v in baseline.items()})))


if __name__=='__main__':main()
