"""Reconcile the complete M78 profile and inspect the exact ORT SiLU implementation."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import onnx

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-packed-final-row-gap-20260925'
REV='2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def read(path):return json.loads(path.read_text(encoding='utf8'))
def pin(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def write(path,value):
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    assert not BASE.exists()
    sources={}
    def closed(name):
        folder=ROOT/'artifacts'/name;proof=read(folder/'closed.json')
        assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
        sources[name]=dict(closure=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
        return read(folder/'analysis.json')
    managed_base=ROOT/'artifacts/parakeet-packed-final-row-profile-closure-amd-20260925'
    managed=closed(managed_base.name)
    native=closed('parakeet-ort-diagnosis-amd-20260924')
    original=closed('parakeet-managed-phase-amd-20260924')
    conv=closed('parakeet-convolution-attribution-20260924')
    graph_review=closed('parakeet-ort-graph-review-20260924')
    projection=read(managed_base/'comparison.json')
    assert projection['managed_closure']==sources[managed_base.name]['closure']
    assert projection['native_closure']==sources['parakeet-ort-diagnosis-amd-20260924']['closure']
    current={r['name']:r for r in managed['phases']['wall']['node_rows'] if r['graph']=='encoder'}
    old={r['name']:r for r in original['phases']['wall']['node_rows'] if r['graph']=='encoder'}
    descriptor=lambda row:{k:v for k,v in row.items() if k not in ['ticks','corpus_seconds']}
    assert len(current)==len(old)==2856 and current.keys()==old.keys()
    for name,row in current.items():assert descriptor(row)==descriptor(old[name]),name
    graph_path=ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder/optimized.onnx'
    assert pin(graph_path)==graph_review['graphs']['encoder']['model']
    graph=onnx.load(graph_path,load_external_data=False).graph
    nodes={n.name:n for n in graph.node}
    native_clock={r['name']:r for r in native['profiles']['encoder']['node_clocks']}
    assert nodes.keys()==native_clock.keys() and len(nodes)==1993
    managed_used=set();native_used=set();partition=[]
    def group(label,ours,theirs):
        ours=set(ours);theirs=set(theirs)
        assert not managed_used.intersection(ours) and not native_used.intersection(theirs),label
        assert all(current[n]['calls']==60 for n in ours)
        assert all(native_clock[n]['calls']==60 for n in theirs)
        managed_used.update(ours);native_used.update(theirs)
        a=sum(current[n]['corpus_seconds'] for n in sorted(ours))
        b=sum(native_clock[n]['exclusive_us'] for n in sorted(theirs))/3e6
        value=dict(group=label,managed_nodes=len(ours),ort_nodes=len(theirs),managed_seconds=a,ort_seconds=b,excess_seconds=a-b,
            managed_members=sorted(ours),ort_members=sorted(theirs))
        partition.append(value);return value
    group('Constant projections including scale',
        [n for r in projection['projections'] for n in r['managed_nodes']], [r['ort_name'] for r in projection['projections']])
    conv_groups=[]
    for kind in ['pointwise_conv1','depthwise_conv','pointwise_conv2']:
        rows=[r for r in conv['convolutions'] if r['kind']==kind]
        assert len(rows)==24
        conv_groups.append(group(kind,[r['name'] for r in rows],[r['name'] for r in rows]))
    # Same stem boundary, with ORT's blocked-layout output reorder included.
    stem_ours=[n for n,r in current.items() if n.startswith('/pre_encode/') and r['op'] in ['Conv','ConvRelu']]
    stem_native=[n.name for n in graph.node if
        (n.op_type=='Conv' and n.name.startswith('/pre_encode/')) or
        (n.op_type=='ReorderOutput' and list(n.output)==['/pre_encode/conv/conv.1_2/Relu_output_0'])]
    assert len(stem_ours)==5 and len(stem_native)==6
    for sequence,accessor in [(stem_ours,lambda name:(current[name]['inputs'],current[name]['outputs'])),
                              (stem_native,lambda name:(list(nodes[name].input),list(nodes[name].output)))]:
        edge='/pre_encode/Unsqueeze_output_0'
        for name in sequence:
            ins,outs=accessor(name);assert ins[0]==edge and len(outs)==1;edge=outs[0]
        assert edge=='/pre_encode/conv/conv.1_2/Relu_output_0'
    group('Convolution stem including fused ReLU and output reorder',stem_ours,stem_native)
    for label,token in [('Attention padding','/self_attn/'),('Convolution padding','/conv/')]:
        ours=[n for n,r in current.items() if r['op']=='Pad' and token in n]
        assert len(ours)==24
        for name in ours:assert nodes[name].op_type=='Pad' and list(nodes[name].output)==current[name]['outputs']
        group(label+' operators',ours,ours)
    activation=[]
    for node in graph.node:
        if node.op_type!='QuickGelu':continue
        assert node.domain=='com.microsoft'
        assert {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}=={'alpha':1.0}
        mul=current[node.name.removesuffix('/QuickGeluFusion/')]
        sigmoid=current[mul['name'].removesuffix('/Mul')+'/Sigmoid']
        assert mul['op']=='Mul' and sigmoid['op']=='Sigmoid'
        assert sigmoid['inputs']==list(node.input) and mul['outputs']==list(node.output)
        assert sorted(mul['inputs'])==sorted([node.input[0],sigmoid['outputs'][0]])
        assert mul['calls']==sigmoid['calls']==native_clock[node.name]['calls']==60
        a=mul['corpus_seconds']+sigmoid['corpus_seconds'];b=native_clock[node.name]['exclusive_us']/3e6
        activation.append(dict(name=node.name,managed_members=[sigmoid['name'],mul['name']],
            family='convolution' if '/conv/' in node.name else 'feed-forward',
            sigmoid_seconds=sigmoid['corpus_seconds'],multiply_seconds=mul['corpus_seconds'],
            managed_seconds=a,ort_seconds=b,excess_seconds=a-b))
    assert len(activation)==72
    for family in ['feed-forward','convolution']:
        rows=[r for r in activation if r['family']==family]
        assert len(rows)==(48 if family=='feed-forward' else 24)
        group(family+' SiLU (sigmoid plus multiply)',[n for r in rows for n in r['managed_members']],[r['name'] for r in rows])
    for label,op in [('Remaining gate sigmoid','Sigmoid'),('Normalization','LayerNormalization'),('Transposes','Transpose')]:
        ours=[n for n,r in current.items() if r['op']==op and n not in managed_used]
        theirs=[n for n,r in native_clock.items() if r['op']==op and n not in native_used]
        group(label,ours,theirs)
    group('All other encoder operators',set(current)-managed_used,set(native_clock)-native_used)
    assert managed_used==set(current) and native_used==set(native_clock)
    a=managed['phases']['wall']['phase_seconds']['encoder']-sum(r['managed_seconds'] for r in partition)
    b=native['phases']['profile']['corpus_phase_seconds']['encoder']-sum(r['ort_seconds'] for r in partition)
    assert a>=0 and b>=0
    partition.append(dict(group='Encoder outside timed operators',managed_seconds=a,ort_seconds=b,excess_seconds=a-b))
    for phase in ['frontend','decoder']:
        a=managed['phases']['wall']['phase_seconds'][phase];b=native['phases']['profile']['corpus_phase_seconds'][phase]
        partition.append(dict(group=phase,managed_seconds=a,ort_seconds=b,excess_seconds=a-b))
    a=managed['phases']['wall']['remainder_seconds']
    native_complete=native['phases']['profile']['corpus_seconds']
    b=native_complete-sum(native['phases']['profile']['corpus_phase_seconds'].values())
    partition.append(dict(group='Outside graph calls',managed_seconds=a,ort_seconds=b,excess_seconds=a-b))
    assert math.isclose(sum(r['managed_seconds'] for r in partition),managed['corpus']['wall'],abs_tol=1e-11)
    assert math.isclose(sum(r['ort_seconds'] for r in partition),native_complete,abs_tol=1e-11)
    module_ours={n for g in conv['modules']['managed']['groups'] for n in g['nodes']}
    module_native={n for g in conv['modules']['ort']['groups'] for n in g['nodes']}
    a=sum(current[n]['corpus_seconds'] for n in sorted(module_ours))
    b=sum(native_clock[n]['exclusive_us'] for n in sorted(module_native))/3e6
    complete_modules=dict(managed_seconds=a,ort_seconds=b,excess_seconds=a-b,managed_nodes=len(module_ours),ort_nodes=len(module_native),
        overlaps_partition=True,do_not_add_to_partition=True)
    BASE.mkdir();native_sources={}
    for name in ['onnxruntime/contrib_ops/cpu/activations.h','onnxruntime/contrib_ops/cpu/activations.cc',
                 'onnxruntime/core/mlas/lib/silu.cpp','onnxruntime/core/mlas/lib/intrinsics/avx512/silu_avx512f.cpp',
                 'onnxruntime/core/mlas/lib/platform.cpp']:
        data=subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),'show',REV+':'+name])
        path=BASE/'native-source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        native_sources[name]=pin(path)
    candidate_source=ROOT/'artifacts/parakeet-packed-final-row-source-20260925/source'
    managed_sources={}
    for name in ['src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs','src/Lokad.Onnx/MathOps.cs','src/Lokad.Onnx/ExecutionOptions.cs']:
        source=candidate_source/name
        assert pin(source)==pin(ROOT/name),name
        managed_sources[name]=pin(source)
    result=dict(passed=True,new_inference_calls=0,sources=sources,source_revision=REV,native_sources=native_sources,
        managed_sources=managed_sources,projection_comparison=pin(managed_base/'comparison.json'),optimized_model=pin(graph_path),
        all_managed_encoder_descriptors_exact=2856,all_native_encoder_nodes=1993,partition=partition,
        complete_convolution_modules=complete_modules,activations=activation,
        activation_total={k:sum(r[k] for r in activation) for k in ['sigmoid_seconds','multiply_seconds','managed_seconds','ort_seconds','excess_seconds']},
        complete_managed=managed['corpus']['wall'],complete_ort=native_complete,
        same_day_native_score=False,profile_clocks_not_application_scores=True,overhead_subtracted=False,
        native_quickgelu_execution_observed=True,native_silu_leaf_is_source_prediction=True,
        selected_change='Vectorize float Sigmoid arithmetic with the existing ExpVector; leave graph fusion and multiply unchanged.',
        release_admitted=False,failed_release_gate='e5-8tok',reviewer=pin(Path(__file__)))
    write(BASE/'analysis.json',result)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),reviewer=pin(Path(__file__)),new_inference_calls=0))
    write(OUT/'remaining-gap-20260925.json',dict(closure=pin(BASE/'closed.json'),**result))
    with (OUT/'activation-groups-20260925.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(activation[0]),lineterminator='\n');writer.writeheader();writer.writerows(activation)
    print(json.dumps(dict(partition=[{k:v for k,v in r.items() if not k.endswith('_members')} for r in partition],
        activations=result['activation_total'],complete_convolution_modules=complete_modules,closure=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
