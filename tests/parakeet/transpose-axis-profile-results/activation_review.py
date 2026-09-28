"""Compare exact SiLU subgraphs, fresh clocks and the pinned ORT implementation."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import onnx

from partition import ROOT, TOOLS, pin, read
from publish import RESULT, native
from run import BASE, APP

REVISION = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def main():
    proof = read(RESULT/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(RESULT/'analysis.json')
    gap = read(RESULT/'analysis.json')
    assert gap['attribution_only'] and not gap['historical_clocks_used']
    profiles = {}
    for role, base in [('managed', BASE), ('native', native.BASE)]:
        assert pin(base/'closed.json') == gap[role+'_profile']
        closed = read(base/'closed.json')
        assert closed['passed'] and closed['analysis'] == pin(base/'analysis.json')
        profiles[role] = read(base/'analysis.json')
    assert 'git-commit-id='+REVISION[:10] in profiles['native']['build_info']
    graph_review = ROOT/'artifacts/parakeet-ort-graph-review-20260924'
    reviewed = read(graph_review/'closed.json')
    assert pin(graph_review/'closed.json')['sha256'] == 'd844dcef266c6ea132c58b68d3e6f2dd04f6a98fed55d4d995badbc21542bf5b'
    assert reviewed['passed'] and reviewed['analysis'] == pin(graph_review/'analysis.json')
    review = read(graph_review/'analysis.json')
    graph_base = ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924'
    graph_file = graph_base/'collected/encoder/optimized.onnx'
    assert pin(graph_file) == review['graphs']['encoder']['model']
    serialized = read(graph_base/'collected/encoder/result.json')
    manifest = read(APP/'collected/manifests/current-parakeet.json')
    assert serialized['source_model'] in [{k:v[k] for k in ['bytes','sha256']} for v in manifest['models'].values()]
    model = onnx.load(str(graph_file), load_external_data=False)
    nodes = {n.name:n for n in model.graph.node}
    ort = profiles['native']['profiles']['encoder']
    assert len(nodes) == len(model.graph.node) and set(nodes) == set(ort['nodes'])
    assert all(n.op_type == ort['nodes'][name]['op'] for name,n in nodes.items())
    managed = {r['name']:r for r in profiles['managed']['phases']['wall']['node_rows'] if r['graph']=='encoder'}
    clocks = {r['name']:r for r in ort['node_clocks']}
    groups, records = [], []
    selected = [('feed-forward SiLU (sigmoid plus multiply)',48), ('convolution SiLU (sigmoid plus multiply)',24)]
    for label, count in selected:
        group = next(r for r in gap['partition'] if r['group']==label)
        used, rows = set(), []
        assert len(group['ort_members']) == count
        for name in group['ort_members']:
            node = nodes[name]
            assert node.op_type == 'QuickGelu' and len(node.input)==len(node.output)==1
            attrs = {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attrs == {'alpha':1.0}
            mul_name = name.removesuffix('/QuickGeluFusion/')
            mul = managed[mul_name]
            sigmoids = [managed[n] for n in group['managed_members'] if managed[n]['op']=='Sigmoid'
                        and managed[n]['outputs'][0] in mul['inputs']]
            assert len(sigmoids)==1 and mul['op']=='Mul'
            sigmoid = sigmoids[0]; members = [sigmoid['name'],mul_name]
            assert not used.intersection(members); used.update(members)
            assert sigmoid['inputs']==list(node.input) and mul['outputs']==list(node.output)
            assert set(mul['inputs']) == {node.input[0],sigmoid['outputs'][0]}
            assert sigmoid['calls']==mul['calls']==clocks[name]['calls']==60
            shapes = [r for r in ort['shapes'] if r['name']==name]
            assert sum(r['calls'] for r in shapes)==80
            assert all(r['inputs']==r['outputs'] and len(r['inputs'])==1 for r in shapes)
            rows.append(dict(native=name,managed_members=members,alpha=1.0,shapes=shapes,
                sigmoid_seconds=sigmoid['corpus_seconds'],multiply_seconds=mul['corpus_seconds'],
                ort_seconds=clocks[name]['exclusive_us']/3e6))
        assert used==set(group['managed_members']) and len(used)==2*count
        a=sum(r['sigmoid_seconds']+r['multiply_seconds'] for r in rows)
        b=sum(r['ort_seconds'] for r in rows)
        assert math.isclose(a,group['managed_seconds'],abs_tol=1e-11)
        assert math.isclose(b,group['ort_seconds'],abs_tol=1e-11)
        groups.append(dict(group=label,nodes=count,sigmoid_seconds=sum(r['sigmoid_seconds'] for r in rows),
            multiply_seconds=sum(r['multiply_seconds'] for r in rows),managed_seconds=a,ort_seconds=b,excess_seconds=a-b))
        records.extend(rows)
    sources = {}
    for name in ['onnxruntime/core/optimizer/quick_gelu_fusion.cc','onnxruntime/contrib_ops/cpu/activations.h',
                 'onnxruntime/core/mlas/lib/silu.cpp','onnxruntime/core/mlas/lib/platform.cpp',
                 'onnxruntime/core/mlas/lib/intrinsics/avx512/silu_avx512f.cpp']:
        data = subprocess.check_output(['git','-C',str(ROOT/'external/onnxruntime'),'show',REVISION+':'+name])
        sources[name] = dict(revision=REVISION,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    value = dict(passed=True,inference_calls=0,partition=pin(RESULT/'closed.json'),
        graph_review=pin(graph_review/'closed.json'),graph=pin(graph_file),sources=sources,
        managed_sources={n:pin(ROOT/n) for n in ['src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs','src/Lokad.Onnx/Zzz.SigmoidRational.cs']},
        groups=groups,records=records,source=pin(Path(__file__)),
        observed_fusion=True,native_leaf_sampled=False,managed_jit_sampled=False,
        cause_not_yet_isolated='The measured aggregate includes arithmetic, two separate graph operations and intermediate storage. Fusion alone is not proven to explain the excess.',
        source_route='QuickGelu alpha=1 calls MlasComputeSilu in 4096-element tasks; platform dispatch can select MlasSiluKernelAvx512F. Exact native leaf and managed generated loop require targeted runtime evidence before a kernel-dependent intervention.')
    with (TOOLS/'activation-breakdown-20260928.json').open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2)
    print(json.dumps(dict(passed=True,groups=groups,native_leaf_sampled=False)))


if __name__ == '__main__': main()
