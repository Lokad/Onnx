"""Promote exact FP32 constants; replace only Conv by explicit double MatMul."""
import copy,collections
import onnx
from onnx import TensorProto as T,helper as h,numpy_helper as nh
from common import np,tensor_array,raw

def lower_conv(node,input_shape,weight,bias_name,add_initializer):
    a={v.name:h.get_attribute_value(v) for v in node.attribute}
    assert a.get('group',1)==1 and a.get('dilations',[1])==[1] and weight.ndim==3
    assert a.get('auto_pad',b'NOTSET')==b'NOTSET'
    batch,channels,length=input_shape;outputs,wchannels,kernel=weight.shape;assert channels==wchannels
    pads=a.get('pads',[0,0]);stride=a.get('strides',[1])[0];count=(length+sum(pads)-kernel)//stride+1
    prefix=node.name+'/float64';created=[]
    def init(suffix,value):
        name=prefix+'/'+suffix;add_initializer(name,np.asarray(value));return name
    pads_name=init('pads',np.array([0,0,pads[0],0,0,pads[1]],dtype=np.int64))
    indices=init('indices',np.arange(count,dtype=np.int64)[:,None]*stride+np.arange(kernel,dtype=np.int64))
    shape=init('shape',np.array([batch,count,channels*kernel],dtype=np.int64))
    matrix=init('weight',weight.reshape(outputs,-1).T.copy())
    previous=node.input[0]
    for suffix,op,extra,attrs in [
        ('pad','Pad',[pads_name],{}),('windows','Gather',[indices],dict(axis=2)),
        ('order','Transpose',[],dict(perm=[0,2,1,3])),('rows','Reshape',[shape],{}),
        ('product','MatMul',[matrix],{}),('bias','Add',[bias_name],{}),
        ('output','Transpose',[],dict(perm=[0,2,1]))]:
        output=node.output[0] if suffix=='output' else prefix+'/'+suffix
        created.append(h.make_node(op,[previous]+extra,[output],name=prefix+'/'+suffix,**attrs));previous=output
    return created,[batch,outputs,count]

def build(original,trace,directory,destination):
    # Removing only outputs must leave the complete original protobuf unchanged.
    first=copy.deepcopy(original);second=copy.deepcopy(trace)
    del first.graph.output[:];del second.graph.output[:]
    assert first.SerializeToString()==second.SerializeToString()
    model=copy.deepcopy(trace);source={i.name:i for i in original.graph.initializer};promoted=[];ledger=[]
    with (destination/'weights.f64').open('xb') as stream:
        def add(name,value,original_value=None):
            assert value.dtype in [np.float64,np.int64]
            if value.dtype==np.int64:promoted.append(nh.from_array(value,name));return
            offset=stream.tell();padding=(-offset)%64;stream.write(bytes(padding));offset+=padding
            value=np.asarray(value,dtype='<f8',order='C');value.tofile(stream)
            tensor=T();tensor.name=name;tensor.data_type=T.DOUBLE;tensor.dims.extend(value.shape);tensor.data_location=T.EXTERNAL
            for k,v in dict(location='weights.f64',offset=str(offset),length=str(value.nbytes)).items():entry=tensor.external_data.add();entry.key=k;entry.value=v
            promoted.append(tensor)
            entry=dict(name=name,shape=list(value.shape),offset=offset,bytes=value.nbytes,sha256=raw(value))
            if original_value is not None:
                assert original_value.dtype==np.float32 and raw(value.astype(np.float32))==raw(original_value)
                entry['original_sha256']=raw(original_value)
            ledger.append(entry)
        for name,tensor in source.items():
            value=tensor_array(tensor,directory)
            add(name,value.astype(np.float64) if value.dtype==np.float32 else value,value if value.dtype==np.float32 else None)
        nodes=[];shape=[1]+[d.dim_value for d in original.graph.input[0].type.tensor_type.shape.dim[1:]];lowered=[]
        assert len(shape)==3 and all(d>0 for d in shape)
        for node in original.graph.node:
            if node.op_type!='Conv':nodes.append(copy.deepcopy(node));continue
            weight=tensor_array(source[node.input[1]],directory).astype(np.float64)
            new,output_shape=lower_conv(node,shape,weight,node.input[2],add);nodes.extend(new)
            lowered.append(dict(name=node.name,input_shape=shape,output_shape=output_shape,new_nodes=[n.name for n in new]));shape=output_shape
        del model.graph.node[:];model.graph.node.extend(nodes);del model.graph.initializer[:];model.graph.initializer.extend(promoted)
    for info in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info):
        if info.type.tensor_type.elem_type==T.FLOAT:info.type.tensor_type.elem_type=T.DOUBLE
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value=1
    onnx.save_model(model,destination/'model.onnx')
    # A path-based checker can resolve external initializers without embedding GBs.
    onnx.checker.check_model(str(destination/'model.onnx'))
    return dict(promotions=ledger,lowered_convolutions=lowered,original_nodes=len(original.graph.node),nodes=len(model.graph.node),outputs=[v.name for v in model.graph.output])

def partition(model,destination):
    """Explicit graph cuts around Erf; every other operation remains in ORT."""
    graph=model.graph;inits={i.name:i for i in graph.initializer}
    known={v.name:v for v in list(graph.input)+list(graph.output)+list(graph.value_info)}
    final={v.name for v in graph.output};nodes=list(graph.node);stages=[];start=0
    def info(name):
        if name in known:return copy.deepcopy(known[name])
        return h.make_tensor_value_info(name,T.DOUBLE,None)
    def add_stage(begin,end):
        selected=nodes[begin:end];produced={n for node in selected for n in node.output}
        needed={n for node in nodes[end:] for n in node.input}
        output=sorted(produced&(needed|final))
        inputs=sorted({n for node in selected for n in node.input}-produced-set(inits))
        constants=sorted({n for node in selected for n in node.input}&set(inits))
        stage=dict(kind='ort',index=len(stages),begin=begin,end=end,inputs=inputs,outputs=output,keep=sorted(needed),nodes=[n.name for n in selected])
        graph=h.make_graph(selected,'double segment',[info(n) for n in inputs],[info(n) for n in output],[inits[n] for n in constants])
        part=h.make_model(graph,opset_imports=model.opset_import,ir_version=model.ir_version)
        stage['file']=f"segment-{len(stages):02}.onnx";onnx.save_model(part,destination/stage['file']);stages.append(stage)
    for i,node in enumerate(nodes):
        if node.op_type!='Erf':continue
        if start<i:add_stage(start,i)
        needed={n for later in nodes[i+1:] for n in later.input}
        stages.append(dict(kind='math_erf',index=len(stages),begin=i,end=i+1,inputs=list(node.input),outputs=list(node.output),keep=sorted(needed),nodes=[node.name]))
        start=i+1
    if start<len(nodes):add_stage(start,len(nodes))
    assert [name for stage in stages for name in stage['nodes']]==[n.name for n in nodes]
    assert len([s for s in stages if s['kind']=='math_erf'])==sum(n.op_type=='Erf' for n in nodes)
    return stages
