"""Prove the exact stem graph and freeze retained inputs, weights and schedule."""
from common import *
import onnx
import subprocess
import torch


def initializer_array(tensor,directory,verified_files):
    """Read only declared slices from the already fully verified model sidecar.

    The retained trace uses intentional hardlinks; ONNX's generic external-file
    loader refuses them. No link-count policy is disabled and no path is inferred
    from an unverified model. Bounds and expected float32 byte lengths are exact.
    """
    if tensor.data_location!=onnx.TensorProto.EXTERNAL:
        return onnx.numpy_helper.to_array(tensor)
    fields={p.key:p.value for p in tensor.external_data}
    assert set(fields)<= {'location','offset','length','checksum'}
    assert fields['location']=='encoder-model.onnx.data'
    path=directory/fields['location'];assert pin(path)==verified_files[rel(path)]
    assert tensor.data_type==onnx.TensorProto.FLOAT
    offset=int(fields.get('offset','0'));length=int(fields['length'])
    assert offset>=0 and length==int(np.prod(tensor.dims))*4 and offset+length<=path.stat().st_size
    with path.open('rb') as f:f.seek(offset);data=f.read(length)
    assert len(data)==length
    return np.frombuffer(data,dtype='<f4').reshape(tuple(tensor.dims)).copy()


def normalized(node):
    attributes={}
    for a in node.attribute:
        value=onnx.helper.get_attribute_value(a)
        if isinstance(value,onnx.TensorProto):
            tensor=onnx.numpy_helper.to_array(value)
            value=dict(dtype=str(tensor.dtype),shape=list(tensor.shape),values=tensor.tolist())
        attributes[a.name]=value
    return dict(name=node.name,op=node.op_type,domain=node.domain,inputs=list(node.input),outputs=list(node.output),attributes=attributes)


def expected_graph():
    rows=[]
    def node(name,op,inputs,attrs=None,output=None):
        output=output or name+'_output_0'
        rows.append(dict(name=name,op=op,domain='',inputs=inputs,outputs=[output],attributes=attrs or {}));return output
    def constant(name,value,output=None):
        a=np.asarray(value,dtype=np.int64)
        return node(name,'Constant',[],dict(value=dict(dtype='int64',shape=list(a.shape),values=a.tolist())),output)
    p='/pre_encode';x=node('/Transpose','Transpose',['audio_signal'],dict(perm=[0,2,1]))
    axis=constant(p+'/Constant_9',[1]);x=node(p+'/Unsqueeze','Unsqueeze',[x,axis])
    relus={0:'conv.1',3:'conv.1_1',6:'conv.1_2'}
    for i in CONVS:
        stride,pad,groups=GEOMETRY[i];k=3 if i in (0,2,5) else 1
        x=node(p+f'/conv/conv.{i}/Conv','Conv',[x,f'pre_encode.conv.{i}.weight',f'pre_encode.conv.{i}.bias'],
               dict(dilations=[1,1],group=groups,kernel_shape=[k,k],pads=[pad]*4,strides=[stride]*2))
        if i in relus:x=node(p+'/conv/'+relus[i]+'/Relu','Relu',[x])
    shape=node(p+'/Shape','Shape',[x]);zero=constant(p+'/Constant_10',0)
    batch=node(p+'/Gather','Gather',[shape,zero],dict(axis=0))
    shape=node(p+'/Shape_1','Shape',[x]);two=constant(p+'/Constant_11',2)
    time=node(p+'/Gather_1','Gather',[shape,two],dict(axis=0))
    x=node(p+'/Transpose','Transpose',[x],dict(perm=[0,2,1,3]))
    zero=constant('Constant_1704',[0],'onnx::Unsqueeze_759');batch=node(p+'/Unsqueeze_1','Unsqueeze',[batch,zero])
    zero=constant('Constant_1706',[0],'onnx::Unsqueeze_761');time=node(p+'/Unsqueeze_2','Unsqueeze',[time,zero])
    minus=constant(p+'/Constant_12',[-1]);shape=node(p+'/Concat','Concat',[batch,time,minus],dict(axis=0))
    x=node(p+'/Reshape','Reshape',[x,shape],dict(allowzero=0))
    x=node(p+'/out/MatMul','MatMul',[x,'onnx::MatMul_6382']);node(p+'/out/Add','Add',['pre_encode.out.bias',x])
    return rows


def main():
    assert not BASE.exists()
    assert np.__version__=='2.2.4' and torch.__version__=='2.11.0+cpu'
    assert pin(PRIOR)['sha256']=='26aeecede5769a5cc1f96774a92e928427dafabd1a2deee312c9e5b683eff025'
    prior=read(PRIOR)
    for path,expected in prior['files'].items():assert pin(ROOT/path)==expected,path
    assert all(absent(i) for i in prior['identities'])
    tests=ROOT/'artifacts/parakeet-stem-reference-tests-v2-20260921.log'
    assert 'Ran 5 tests' in tests.read_text() and '\nOK' in tests.read_text()
    old=read(TRACE/'manifest.json');original=ROOT/old['models']['plain']
    model=onnx.load(original,load_external_data=False);needed={'/pre_encode/out/Add_output_0'};selected=[]
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            selected.append(node);needed.difference_update(node.output);needed.update(node.input)
    selected.reverse();assert [normalized(n) for n in selected]==expected_graph()
    names={f'conv{i}.{kind}':f'pre_encode.conv.{i}.{kind}' for i in CONVS for kind in ('weight','bias')}
    names.update({'projection.weight':'onnx::MatMul_6382','projection.bias':'pre_encode.out.bias'})
    assert needed=={'audio_signal',*names.values()}
    BASE.mkdir();(BASE/'weights').mkdir();files={}
    def bind(path):files[rel(path)]=pin(path);return rel(path)
    bind(PRIOR);bind(tests);bind(ROOT/'artifacts/parakeet-stem-reference-tests-20260921.log');bind(TRACE/'manifest.json');bind(original)
    for path,expected in old['files'].items():
        if path.startswith('models/parakeet-tdt-0.6b-v3/encoder-model.onnx'):
            assert pin(ROOT/path)==expected;bind(ROOT/path)
    tensors={t.name:t for t in model.graph.initializer};weights={}
    for key,name in names.items():
        value=initializer_array(tensors[name],original.parent,files)
        assert value.dtype==np.float32 and np.isfinite(value).all()
        path=BASE/'weights'/(key+'.npy');np.save(path,value,allow_pickle=False)
        weights[key]=dict(file=bind(path),shape=list(value.shape),source=name,raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    assert weights['projection.weight']['shape']==[4096,1024] and weights['projection.bias']['shape']==[1024]
    for i in CONVS:
        assert weights[f'conv{i}.weight']['shape']==([256,1,3,3] if i in (0,2,5) else [256,256,1,1])
        assert weights[f'conv{i}.bias']['shape']==[256]
    inputs={};stems={}
    for kind in ('native','managed'):
        path=ROOT/old['inputs'][kind]['features'];assert pin(path)==old['files'][rel(path)]
        x=np.load(path,allow_pickle=False);assert x.dtype==np.float32 and x.shape==(1,128,586)
        inputs[kind]=bind(path)
        for engine,folder in [('managed',TRACE/'outputs'/('managed-'+kind+'-trace')),('native',DYNAMIC/'outputs'/kind)]:
            result=read(folder/'result.json');bind(folder/'result.json')
            r=next(r for r in result['outputs'] if r['name']=='/pre_encode/out/Add_output_0')
            assert r['shape']==[1,74,1024] and r['dtype']=='float32'
            load_array(folder/r['file'],r);stems[engine+'-'+kind]=dict(r)
            stems[engine+'-'+kind]['file']=bind(folder/r['file'])
    write(BASE/'graph.json',dict(nodes=[normalized(n) for n in selected],nodes_sha256=hashlib.sha256(b''.join(n.SerializeToString() for n in selected)).hexdigest()))
    bind(BASE/'graph.json')
    for path in Path(__file__).parent.iterdir():
        if path.is_file():bind(path)
    helper=ROOT/'tests/pyannote/filterbank-reference/common.py';bind(helper)
    helpers=reference_helpers();assert helpers.openblas_threads()==1
    libraries=helpers.libraries(psutil.Process())
    for path,expected in libraries.items():assert pin(path)==expected;bind(path)
    for path in (np.__file__,torch.__file__,torch.nn.functional.__file__):bind(path)
    write(BASE/'manifest.json',dict(protocol=PROTOCOL,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          limits=LIMITS,jobs=JOBS,stages=STAGES,reference_limit=REF_LIMIT,original_limit=1e-4,inputs=inputs,weights=weights,
          retained_stems=stems,files=files,numerical_libraries=libraries,numpy=np.__version__,torch=torch.__version__,onnx=onnx.__version__))
    print(json.dumps(dict(manifest=pin(BASE/'manifest.json'),weights=len(weights),nodes=len(selected),jobs=len(JOBS))))


if __name__=='__main__':main()
