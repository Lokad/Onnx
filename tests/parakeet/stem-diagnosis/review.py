"""Explain the exact M78 stem using retained graphs, clocks and pinned source."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import onnx

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-stem-diagnosis-20260925'
REV='2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
VOICE='2b1138fe6f5d085e3749f6867d1603b1131ff029'
SOURCE=ROOT/'artifacts/parakeet-packed-final-row-source-20260925'


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf8'))
def write(path,value):
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def attributes(node):
    return {a.name:(v.decode() if isinstance(v,bytes) else v) for a in node.attribute for v in [onnx.helper.get_attribute_value(a)]}


def spatial(c,kh,kw,m,oh,ow,group):
    columns=oh*ow;reduction=c*kh*kw;block=columns
    if reduction*columns>256*1024//4:
        fit=256*1024//((reduction+m)*4);block=min(columns,max(32,fit//32*32))
    tiled=block<columns;blocks=(columns+block-1)//block
    products=blocks*group
    return dict(columns=columns,reduction=reduction,block_columns=block,blocks=blocks,tiled=tiled,
        scratch_requested_bytes=(reduction*block+(m*block if tiled else 0))*4,
        patch_values_written=reduction*columns,matrix_calls=products,tensor_views=3*products,
        matrix_m=m//group,matrix_k=reduction//group,full_matrix_n=block,
        final_matrix_n=columns-(blocks-1)*block)


def main():
    assert not BASE.exists();sources={}
    def closed(name):
        base=ROOT/'artifacts'/name;proof=read(base/'closed.json')
        assert proof['passed'] and proof['analysis']==pin(base/'analysis.json')
        sources[name]=dict(closure=pin(base/'closed.json'),analysis=pin(base/'analysis.json'))
        return read(base/'analysis.json')
    gap=closed('parakeet-packed-final-row-gap-20260925')
    managed=closed('parakeet-packed-final-row-profile-closure-amd-20260925')
    native=closed('parakeet-ort-diagnosis-amd-20260924')
    graphs=closed('parakeet-ort-graph-review-20260924')
    prior=closed('parakeet-convolution-attribution-20260924')
    screen=closed('parakeet-vector-sigmoid-screen-amd-20260925');assert not screen['admitted']
    stem=next(r for r in gap['partition'] if r['group'].startswith('Convolution stem'))
    opath=ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    npath=ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder/optimized.onnx'
    assert pin(opath)==prior['original_model'] and pin(npath)==graphs['graphs']['encoder']['model']==prior['optimized_model']
    original=onnx.load(opath,load_external_data=False).graph;optimized=onnx.load(npath,load_external_data=False).graph
    originals={n.name:n for n in original.node};nodes={n.name:n for n in optimized.node}
    init={t.name:t for t in original.initializer};packed={t.name:t for t in optimized.initializer}
    ours={r['name']:r for r in managed['phases']['wall']['node_rows'] if r['graph']=='encoder'}
    observed=native['profiles']['encoder'];clocks={r['name']:r for r in observed['node_clocks']}
    assert len(ours)==2856 and len(nodes)==len(clocks)==1993
    native_stem=[n for n in optimized.node if n.name in stem['ort_members']]
    assert len(native_stem)==6 and native_stem[-1].op_type=='ReorderOutput'
    rows=[];shape_rows=[];edge='/pre_encode/Unsqueeze_output_0';native_edge=edge
    for index,(part,n) in enumerate(zip([0,2,3,5,6],native_stem[:5],strict=True)):
        name=f'/pre_encode/conv/conv.{part}/Conv';old=originals[name];m=ours[name]
        assert name in stem['managed_members'] and old.op_type=='Conv' and m['op'] in ['Conv','ConvRelu']
        assert old.input[0]==m['inputs'][0]==edge and n.input[0]==native_edge
        assert list(old.input)==m['inputs'] and n.domain=='com.microsoft.nchwc' and n.op_type=='Conv'
        oa=attributes(old);na=attributes(n);oa.setdefault('auto_pad','NOTSET')
        assert {k:v for k,v in na.items() if k!='activation'}==oa
        if m['op']=='ConvRelu':
            relu=next(r for r in original.node if r.op_type=='Relu' and list(r.input)==list(old.output))
            assert list(relu.output)==m['outputs'] and na['activation']=='Relu'
        else:assert list(old.output)==m['outputs'] and 'activation' not in na
        assert list(init[old.input[1]].dims)==list(packed[n.input[1]].dims)==m['constant_inputs'][1]['dims']
        assert old.input[2]==n.input[2] and list(init[old.input[2]].dims)==list(packed[n.input[2]].dims)==[256]
        assert all(t.data_type==onnx.TensorProto.FLOAT for t in [init[old.input[1]],packed[n.input[1]],init[old.input[2]],packed[n.input[2]]])
        assert m['calls']==clocks[n.name]['calls']==60
        shapes=[r for r in observed['shapes'] if r['name']==n.name]
        assert sum(r['calls'] for r in shapes)==80 and all(r['calls']%4==0 for r in shapes)
        assert len(shapes)==(19 if part==6 else 20)
        depthwise=oa['group']==256;pointwise=oa['kernel_shape']==[1,1]
        totals=Counter();minimum=math.inf;maximum=0
        for s in shapes:
            assert all(list(v)==['float'] for v in s['inputs']+s['outputs'])
            x,w,b=[v['float'] for v in s['inputs']];y=s['outputs'][0]['float']
            assert len(x)==len(w)==len(y)==4 and x[0]==y[0]==1 and y[1]==w[0]==256 and b==[256]
            assert x[1]==w[1]*oa['group'] and w==list(init[old.input[1]].dims)
            assert oa['dilations']==[1,1]
            assert all(y[2+i]==(x[2+i]+oa['pads'][i]+oa['pads'][i+2]-w[2+i])//oa['strides'][i]+1 for i in range(2))
            weight=s['calls']//4
            if pointwise:
                costs=dict(matrix_calls=1,tensor_views=3,patch_values_written=0,scratch_requested_bytes=0,
                    matrix_m=256,matrix_k=256,full_matrix_n=y[2]*y[3],final_matrix_n=y[2]*y[3],blocks=1,block_columns=y[2]*y[3],tiled=False)
            else:
                costs=spatial(x[1],w[2],w[3],y[1],y[2],y[3],oa['group'])
                assert costs['tiled']
                if depthwise:assert costs['block_columns']==32 and costs['matrix_m']==1 and costs['matrix_k']==9
            for key in ['matrix_calls','tensor_views','patch_values_written']:totals[key]+=weight*costs[key]
            minimum=min(minimum,costs['scratch_requested_bytes']);maximum=max(maximum,costs['scratch_requested_bytes'])
            shape_rows.append(dict(family='stem',name=name,native_name=n.name,input=x,weight_shape=w,output=y,frequency=weight,attributes=oa,predicted=costs))
        rows.append(dict(managed=name,native=n.name,managed_op=m['op'],native_domain=n.domain,attributes=na,
            managed_seconds=m['corpus_seconds'],ort_seconds=clocks[n.name]['exclusive_us']/3e6,
            excess_seconds=m['corpus_seconds']-clocks[n.name]['exclusive_us']/3e6,shapes=len(shapes),
            managed_route='RunPointwiseBatchesFloat' if pointwise else 'RunTiledBatchFloat',
            native_route='MLAS_NCHWC_CONV_DEPTHWISE_ALGORITHM' if depthwise else 'MLAS_NCHWC_CONV_POINTWISE_ALGORITHM' if pointwise else 'MLAS_NCHWC_CONV_NCHW_ALGORITHM',
            native_filter_layout='OIHWBiBo' if pointwise else 'OIHWBo',
            managed_convolution_weight_packing_eligible=False,predicted_counts=dict(totals),
            scratch_requested_minimum=minimum,scratch_requested_maximum=maximum))
        edge=m['outputs'][0];native_edge=n.output[0]
    reorder=native_stem[-1]
    assert list(reorder.input)==[native_edge] and list(reorder.output)==[edge] and edge=='/pre_encode/conv/conv.1_2/Relu_output_0'
    assert attributes(reorder)==dict(channels=256,channels_last=0) and clocks[reorder.name]['calls']==60
    reorder_shapes=[r for r in observed['shapes'] if r['name']==reorder.name]
    assert sum(r['calls'] for r in reorder_shapes)==80 and all(r['inputs']==r['outputs'] for r in reorder_shapes)
    reorder_seconds=clocks[reorder.name]['exclusive_us']/3e6
    assert math.isclose(sum(r['managed_seconds'] for r in rows),stem['managed_seconds'],abs_tol=1e-12)
    assert math.isclose(sum(r['ort_seconds'] for r in rows)+reorder_seconds,stem['ort_seconds'],abs_tol=1e-12)
    depths=[r for r in rows if r['attributes']['group']==256];assert len(depths)==2
    depthwise={k:sum(r[k] for r in depths) for k in ['managed_seconds','ort_seconds','excess_seconds']}
    depthwise['fraction_of_stem_excess']=depthwise['excess_seconds']/stem['excess_seconds']
    module_seconds=0.;module_native=0.;module_counts=Counter()
    for layer in range(24):
        name=f'/layers.{layer}/conv/depthwise_conv/Conv';n=nodes[name];m=ours[name];old=originals[name]
        assert m['op']==n.op_type==old.op_type=='Conv' and not n.domain
        assert list(n.input)==list(old.input)==m['inputs'] and list(n.output)==list(old.output)==m['outputs']
        attrs=attributes(n);oa=attributes(old);oa.setdefault('auto_pad','NOTSET')
        assert attrs==oa==dict(auto_pad='NOTSET',dilations=[1],group=1024,strides=[1],kernel_shape=[9],pads=[0,0])
        assert m['calls']==clocks[name]['calls']==60
        seen=Counter()
        for s in observed['shapes']:
            if s['name']!=name:continue
            assert all(list(v)==['float'] for v in s['inputs']+s['outputs'])
            x,w,b=[v['float'] for v in s['inputs']];y=s['outputs'][0]['float'];t=y[-1]
            assert x==[1,1024,t+8] and w==[1024,1,9] and b==[1024] and y==[1,1024,t]
            assert s['calls']%4==0;frequency=s['calls']//4;seen[t]+=frequency
            costs=spatial(1024,1,9,1024,1,t,1024);assert costs['block_columns']==32 and costs['scratch_requested_bytes']==1310720
            for key in ['matrix_calls','tensor_views','patch_values_written']:module_counts[key]+=frequency*costs[key]
            shape_rows.append(dict(family='modules',name=name,native_name=name,input=x,weight_shape=w,output=y,frequency=frequency,attributes=attrs,predicted=costs))
        assert seen==Counter({int(k):v for k,v in prior['frames'].items()})
        module_seconds+=m['corpus_seconds'];module_native+=clocks[name]['exclusive_us']/3e6
    assert module_counts['matrix_calls']==2236416
    module=dict(nodes=24,managed_seconds=module_seconds,ort_seconds=module_native,
        excess_seconds=module_seconds-module_native,predicted_counts=dict(module_counts),
        native_route='MlasConvAlgorithmExpandThenGemmSegmented -> MlasSgemmKernelM1Avx; source prediction, not NCHWc')
    BASE.mkdir();native_sources={};managed_sources={}
    for name in ['onnxruntime/contrib_ops/cpu/nchwc_ops.cc','onnxruntime/core/optimizer/nchwc_transformer.cc',
        'onnxruntime/core/mlas/lib/snchwc.cpp','onnxruntime/core/mlas/lib/reorder.cpp','onnxruntime/core/mlas/lib/platform.cpp',
        'onnxruntime/core/mlas/lib/x86_64/SconvKernelAvx512F.S','onnxruntime/core/mlas/lib/x86_64/SconvKernelCommon.h',
        'onnxruntime/core/providers/cpu/nn/conv.cc','onnxruntime/core/mlas/lib/convolve.cpp','onnxruntime/core/mlas/lib/sgemm.cpp']:
        data=subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),'show',REV+':'+name])
        path=BASE/'native-source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);native_sources[name]=pin(path)
    receipt=read(SOURCE/'prepared.json');assert len(receipt['source'])==433
    for name in ['CPUExecutionProvider.ConvPool.cs','CPUExecutionProvider.Fusion.cs','TensorOps.ConvPool.cs','TensorOps.ConvBlocked.cs',
        'Zzz.ConvDirectOutput.cs','Zzz.ConvPortableRows.cs','GraphConvPacking.cs','TensorOps.MatMul.cs','Zzz.WideProjectionEntry.cs',
        'TensorExecutionOptions.cs','AblationSwitches.cs']:
        relative='src/Lokad.Onnx/'+name;path=SOURCE/'source'/relative
        assert pin(path)==receipt['source'][relative];managed_sources[relative]=pin(path)
    voice=subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT),'show',VOICE+':src/Lokad.Onnx/TensorOps.ConvPool.cs'])
    (BASE/'voice-ConvPool.cs.txt').write_bytes(voice)
    combined_counts={k:sum(r['predicted_counts'][k] for r in depths)+module_counts[k] for k in module_counts}
    result=dict(passed=True,new_inference_calls=0,sources=sources,source_revision=REV,native_sources=native_sources,
        managed_sources=managed_sources,source_snapshot=pin(SOURCE/'prepared.json'),original_model=pin(opath),optimized_model=pin(npath),
        native_external_weights_retired=not (npath.parent/'weights.bin').exists(),filter_reorder_bytes_revalidated=False,
        rows=rows,shapes=shape_rows,reorder_seconds=reorder_seconds,stem=stem,stem_depthwise=depthwise,modules=module,
        combined_depthwise=dict(nodes=26,managed_seconds=depthwise['managed_seconds']+module_seconds,
            ort_seconds=depthwise['ort_seconds']+module_native,excess_seconds=depthwise['excess_seconds']+module['excess_seconds'],
            predicted_counts=combined_counts),
        voice=dict(revision=VOICE,source=pin(BASE/'voice-ConvPool.cs.txt'),imported=False,
            cautions=['Bias-seeded accumulation differs from current sum-then-bias.',
                      'Stride-two 2D case stays scalar in the branch.',
                      'Do not import compiler annotations, parallel scheduling, fusion or uninitialized pooling.']),
        executed_native_node_routes_observed=True,internal_native_leaf_observed=False,managed_inner_routes_observed=False,
        counts_are_source_predictions=True,times_are_dated_profile_diagnostics=True,release_admitted=False,
        next_action='Observe actual M78 depthwise tiling, tensor views and matrix routes for all corpus shapes before selecting an implementation.',
        reviewer=pin(Path(__file__)))
    write(BASE/'analysis.json',result)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),reviewer=pin(Path(__file__)),new_inference_calls=0))
    write(TOOLS/'diagnosis-20260925.json',dict(closure=pin(BASE/'closed.json'),**result))
    with (TOOLS/'stem-20260925.csv').open('x',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['managed','native','managed_seconds','ort_seconds','excess_seconds','matrix_calls_predicted','tensor_views_predicted'])
        for r in rows:writer.writerow([r['managed'],r['native'],r['managed_seconds'],r['ort_seconds'],r['excess_seconds'],r['predicted_counts']['matrix_calls'],r['predicted_counts']['tensor_views']])
        writer.writerow(['','ReorderOutput',0,reorder_seconds,-reorder_seconds,0,0])
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),stem_depthwise=depthwise,modules=module,combined=result['combined_depthwise'],rows=rows)))


if __name__=='__main__':main()
