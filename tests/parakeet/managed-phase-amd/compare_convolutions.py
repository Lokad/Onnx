"""Reconcile all 24 convolution modules using retained graphs and clocks only."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import onnx

from run import ROOT, BASE, pin, read, write


def closed(folder):
    proof=read(folder/'closed.json')
    assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
    return read(folder/'analysis.json')


def attributes(node):
    return {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}


def trace_modules(nodes, initializers):
    producers={edge:node for node in nodes.values() for edge in node['outputs'] if edge}
    assert len(producers)==sum(bool(edge) for node in nodes.values() for edge in node['outputs'])
    groups=[];union={};memberships=defaultdict(list)
    for layer in range(24):
        prefix=f'/layers.{layer}/conv/'
        boundary={f'/layers.{layer}/norm_conv/LayerNormalization_output_0','/Not_1_output_0'}
        found={};leaves=set();visiting=set()
        def visit(edge):
            if not edge:return
            if edge in boundary or edge in initializers:
                leaves.add(edge);return
            assert edge in producers,edge
            node=producers[edge];name=node['name']
            assert name not in visiting,name
            if name in found:return
            visiting.add(name)
            for source in node['inputs']:visit(source)
            visiting.remove(name);found[name]=node
        visit(prefix+'Transpose_1_output_0')
        assert leaves & boundary==boundary
        for name,node in found.items():
            assert node['calls']==60,(name,node['calls'])
            union[name]=node;memberships[name].append(layer)
        groups.append(dict(layer=layer,boundary=sorted(boundary),output=prefix+'Transpose_1_output_0',
            nodes=sorted(found),initializer_leaves=sorted(leaves-boundary)))
    operators=defaultdict(lambda:dict(nodes=0,seconds=0.))
    for node in union.values():
        operators[node['op']]['nodes']+=1;operators[node['op']]['seconds']+=node['seconds']
    shared=[dict(name=name,layers=layers,seconds=union[name]['seconds'])
            for name,layers in memberships.items() if len(layers)>1]
    return dict(groups=groups,unique_nodes=len(union),seconds=sum(r['seconds'] for r in union.values()),
        operators=dict(operators),shared_nodes=shared,shared_seconds=sum(r['seconds'] for r in shared)),union


def main():
    managed=closed(BASE)
    native_base=ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924';native=closed(native_base)
    review_base=ROOT/'artifacts/parakeet-ort-graph-review-20260924';review=closed(review_base)
    optimized_path=ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder/optimized.onnx'
    assert pin(optimized_path)==review['graphs']['encoder']['model']
    original_path=ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    manifest=read(ROOT/'artifacts/parakeet-prepared-recurrence-app-amd-20260924/collected/manifests/current-parakeet.json')
    assert pin(original_path)=={k:manifest['models']['encoder-model.onnx'][k] for k in ['bytes','sha256']}
    original=onnx.load(original_path,load_external_data=False).graph
    optimized=onnx.load(optimized_path,load_external_data=False).graph
    originals={n.name:n for n in original.node};natives={n.name:n for n in optimized.node}
    original_constants={t.name:t for t in original.initializer};native_constants={t.name:t for t in optimized.initializer}
    observations=native['profiles']['encoder'];clocks={n['name']:n for n in observations['node_clocks']}
    assert set(clocks)==set(natives)
    managed_nodes={n['name']:dict(n,seconds=n['corpus_seconds']) for n in managed['phases']['wall']['node_rows'] if n['graph']=='encoder'}
    native_nodes={name:dict(name=name,op=n.op_type,inputs=list(n.input),outputs=list(n.output),
        seconds=clocks[name]['exclusive_us']/3e6,calls=clocks[name]['calls']) for name,n in natives.items()}
    frames=Counter(case['expected']['encoded_frames'] for case in manifest['cases'])
    assert len(frames)==19 and sum(frames.values())==20
    rows=[]
    for layer in range(24):
        for kind in ['pointwise_conv1','depthwise_conv','pointwise_conv2']:
            name=f'/layers.{layer}/conv/{kind}/Conv';old=originals[name];new=natives[name];observed=managed_nodes[name]
            assert old.op_type==new.op_type==observed['op']=='Conv'
            assert list(old.input)==list(new.input)==observed['inputs']
            assert list(old.output)==list(new.output)==observed['outputs']
            # ORT serialization makes the default explicit. Its pinned
            # ConvAttributes initializes missing auto_pad to NOTSET.
            old_attributes=attributes(old);new_attributes=attributes(new)
            assert 'auto_pad' not in old_attributes and new_attributes['auto_pad']==b'NOTSET'
            assert dict(old_attributes,auto_pad=b'NOTSET')==new_attributes
            depthwise=kind=='depthwise_conv';filters=2048 if kind=='pointwise_conv1' else 1024
            expected=dict(auto_pad=b'NOTSET',dilations=[1],group=1024 if depthwise else 1,
                strides=[1],kernel_shape=[9] if depthwise else [1],pads=[0,0])
            assert attributes(new)==expected,(name,attributes(new))
            weights=[1024,1,9] if depthwise else [filters,1024,1]
            for index,shape in [(1,weights)]+([(2,[1024])] if depthwise else []):
                edge=old.input[index];item=observed['constant_inputs'][index]
                assert item['name']==edge and item['dims']==shape and item['type']=='Float'
                assert list(original_constants[edge].dims)==list(native_constants[edge].dims)==shape
                assert original_constants[edge].data_type==native_constants[edge].data_type==onnx.TensorProto.FLOAT
            seen=Counter()
            for shape in observations['shapes']:
                if shape['name']!=name:continue
                ins=[next(iter(s.values())) for s in shape['inputs']]
                outs=[next(iter(s.values())) for s in shape['outputs']]
                assert len(outs)==1 and outs[0][:2]==[1,filters]
                t=outs[0][2]
                assert ins==[[1,1024,t+8 if depthwise else t],weights]+([[1024]] if depthwise else [])
                seen[t]+=shape['calls']
            assert seen==Counter({t:count*4 for t,count in frames.items()})
            assert observed['calls']==clocks[name]['calls']==60
            seconds=observed['seconds'];reference=native_nodes[name]['seconds']
            rows.append(dict(layer=layer,kind=kind,name=name,managed_seconds=seconds,ort_seconds=reference,
                excess_seconds=seconds-reference,measured_calls=60,shape_observations=len(seen)))
    module_reports={};unions={}
    for label,nodes,initializers in [('managed',managed_nodes,original_constants),('ort',native_nodes,native_constants)]:
        module_reports[label],unions[label]=trace_modules(nodes,initializers)
    assert [g['boundary'] for g in module_reports['managed']['groups']]==[g['boundary'] for g in module_reports['ort']['groups']]
    assert [g['output'] for g in module_reports['managed']['groups']]==[g['output'] for g in module_reports['ort']['groups']]
    for label in unions:
        assert {name for name,n in unions[label].items() if n['op']=='Conv'}=={r['name'] for r in rows}
    grouped=[]
    for kind in ['pointwise_conv1','depthwise_conv','pointwise_conv2']:
        values=[r for r in rows if r['kind']==kind];assert len(values)==24
        current=sum(r['managed_seconds'] for r in values);reference=sum(r['ort_seconds'] for r in values)
        grouped.append(dict(kind=kind,nodes=24,managed_seconds=current,ort_seconds=reference,excess_seconds=current-reference))
    revision=review['ort_revision'];native_sources={}
    for name in ['onnxruntime/core/providers/cpu/nn/conv.cc','onnxruntime/core/providers/cpu/nn/conv_attributes.h',
                 'onnxruntime/core/providers/common.h','onnxruntime/core/mlas/lib/convolve.cpp',
                 'onnxruntime/core/mlas/lib/sgemm.cpp','onnxruntime/core/mlas/lib/platform.cpp']:
        data=subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),'show',revision+':'+name])
        native_sources[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    sources={name:pin(ROOT/name) for name in ['src/Lokad.Onnx/CPUExecutionProvider.ConvPool.cs',
        'src/Lokad.Onnx/TensorOps.ConvPool.cs','src/Lokad.Onnx/TensorOps.ConvBlocked.cs',
        'src/Lokad.Onnx/Zzz.ConvDirectOutput.cs','src/Lokad.Onnx/Zzz.ConvPortableRows.cs']}
    result=dict(passed=True,managed_closure=pin(BASE/'closed.json'),native_closure=pin(native_base/'closed.json'),
        graph_review=pin(review_base/'closed.json'),original_model=pin(original_path),optimized_model=pin(optimized_path),
        ort_revision=revision,native_sources=native_sources,managed_sources=sources,groups=grouped,
        modules=module_reports,convolutions=rows,frames=dict(sorted(frames.items())),
        matched_by_edges_attributes_parameters_and_runtime_shapes=True,shared_ancestors_counted_once=True,
        attribute_normalization='Only the omitted original auto_pad equals serialized NOTSET, as initialized in the pinned ConvAttributes source.',
        source_dispatch_is_prediction=True,per_node_native_leaf_observed=False,managed_runtime_route_observed=False,
        profiler_clocks_are_diagnostic=True,new_inference_calls=0,new_optimization_selected=False,
        caveat='Inclusive ancestor unions contain shared constant/mask work once; do not add these totals to other attribution groups without removing overlap.')
    out=ROOT/'artifacts/parakeet-convolution-attribution-20260924';assert not out.exists();out.mkdir()
    write(out/'analysis.json',result)
    write(out/'closed.json',dict(passed=True,analysis=pin(out/'analysis.json'),auditor=pin(__file__),new_inference_calls=0))
    report=ROOT/'tests/parakeet/managed-phase-results'
    with (report/'convolutions-20260924.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (report/'convolution-observations-20260924.json').open('x',encoding='utf8') as f:
        json.dump(dict(closure=pin(out/'closed.json'),**{k:v for k,v in result.items() if k not in ['convolutions','modules']},
            modules={k:{key:value for key,value in r.items() if key!='groups'} for k,r in module_reports.items()}),f,indent=2);f.write('\n')
    print(json.dumps(dict(closed=pin(out/'closed.json'),groups=grouped,
        modules={k:{key:r[key] for key in ['unique_nodes','seconds','shared_seconds']} for k,r in module_reports.items()})))


if __name__=='__main__':main()
