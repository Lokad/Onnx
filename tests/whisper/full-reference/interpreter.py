"""Direct, unfused float64 interpretation of the original encoder operators."""
import collections
import onnx
from common import np,packages,tensor_array,OPS
packages()
from scipy.special import erf

def attributes(node):return {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}

def conv(x,w,bias,attrs):
    assert x.ndim==w.ndim==3 and x.shape[1]==w.shape[1]
    assert attrs.get('group',1)==1 and attrs.get('dilations',[1])==[1]
    assert attrs.get('auto_pad',b'NOTSET')==b'NOTSET'
    pads=attrs.get('pads',[0,0]);stride=attrs.get('strides',[1])[0]
    assert len(pads)==2 and stride>0 and all(p>=0 for p in pads)
    padded=np.pad(x,((0,0),(0,0),tuple(pads)))
    windows=np.lib.stride_tricks.sliding_window_view(padded,w.shape[2],axis=2)[:,:,::stride,:]
    values=windows.transpose(0,2,1,3).reshape(x.shape[0],windows.shape[2],-1)
    output=values@w.reshape(w.shape[0],-1).T
    if bias is not None:output+=bias
    return output.transpose(0,2,1)

def execute(node,inputs):
    a=attributes(node);op=node.op_type
    assert op in OPS and node.domain in ['', 'ai.onnx'] and len(node.output)==1
    if op=='Add':return inputs[0]+inputs[1]
    if op=='Sub':return inputs[0]-inputs[1]
    if op=='Mul':return inputs[0]*inputs[1]
    if op=='Div':return inputs[0]/inputs[1]
    if op=='Pow':return np.power(inputs[0],inputs[1])
    if op=='MatMul':return np.matmul(inputs[0],inputs[1])
    if op=='Sqrt':return np.sqrt(inputs[0])
    if op=='Erf':return erf(inputs[0])
    if op=='ReduceMean':return np.mean(inputs[0],axis=tuple(a['axes']) if 'axes' in a else None,keepdims=bool(a.get('keepdims',1)),dtype=np.float64)
    if op=='Transpose':return np.transpose(inputs[0],a.get('perm'))
    if op=='Reshape':
        shape=inputs[1].astype(np.int64).copy()
        if not a.get('allowzero',0):
            for i,d in enumerate(shape):
                if d==0:shape[i]=inputs[0].shape[i]
        return inputs[0].reshape(tuple(shape))
    if op=='Softmax':
        axis=a.get('axis',-1);v=inputs[0]-np.max(inputs[0],axis=axis,keepdims=True);v=np.exp(v)
        return v/np.sum(v,axis=axis,keepdims=True,dtype=np.float64)
    if op=='Conv':return conv(inputs[0],inputs[1],inputs[2] if len(inputs)>2 else None,a)
    raise AssertionError(op)

def run(model,directory,features,capture):
    assert len(model.graph.input)==1 and features.dtype==np.float64
    initializers={i.name:i for i in model.graph.initializer};uses=collections.Counter(name for n in model.graph.node for name in n.input)
    values={model.graph.input[0].name:features};records=[]
    for index,node in enumerate(model.graph.node):
        inputs=[]
        for name in node.input:
            if name in values:value=values[name]
            else:
                assert name in initializers,name
                value=tensor_array(initializers[name],directory)
                if value.dtype==np.float32:value=value.astype(np.float64)
            inputs.append(value)
        output=execute(node,inputs);assert output.dtype==np.float64 and np.isfinite(output).all(),node.name
        records.append(dict(index=index,name=node.name,op=node.op_type,shape=list(output.shape)))
        capture(node.output[0],output)
        if uses[node.output[0]]:values[node.output[0]]=output
        for name in node.input:
            uses[name]-=1
            if uses[name]==0:values.pop(name,None)
        del inputs,output,value
    assert not values and all(v==0 for v in uses.values())
    return records
