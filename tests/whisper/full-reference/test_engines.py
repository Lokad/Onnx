"""Independent ONNX semantics and complete small-network reference tests."""
import copy,tempfile,unittest
from pathlib import Path
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as T
from onnx.reference import ReferenceEvaluator
from common import np,packages,OPS,tensor_array
from interpreter import execute,run
from promote import build,partition
from native import run as native_run,double_erf
from onnx.reference.ops._op import OpRunUnaryNum
packages()
import onnxruntime as ort

class Erf(OpRunUnaryNum):
    # ONNX1.22 default uses otypes=['f'], silently rounding through float32.
    def _run(self,x):return (double_erf(x),)

def fixture():
    rng=np.random.default_rng(71023);nodes=[];inits=[];infos=[]
    def init(name,value):inits.append(nh.from_array(np.asarray(value),name));return name
    def add(op,inputs,name,shape,**attrs):
        nodes.append(h.make_node(op,inputs,[name],name=name,**attrs));infos.append(h.make_tensor_value_info(name,T.FLOAT,shape));return name
    init('cw',(rng.standard_normal((4,2,3))*.1).astype('f4'));init('cb',np.array([.1,-.2,.3,-.4],dtype='f4'))
    init('one',np.array(1,dtype='f4'));init('half',np.array(.5,dtype='f4'));init('root2',np.array(np.sqrt(2),dtype='f4'))
    init('two',np.array(2,dtype='f4'));init('epsilon',np.array(1e-5,dtype='f4'))
    init('lnw',np.array([1.1,.7,1.3,.9],dtype='f4'));init('lnb',np.array([.1,.2,-.1,-.2],dtype='f4'))
    def gelu(x,shape,prefix):
        d=add('Div',[x,'root2'],prefix+'d',shape);e=add('Erf',[d],prefix+'e',shape)
        a=add('Add',[e,'one'],prefix+'a',shape);m=add('Mul',[x,a],prefix+'m',shape)
        return add('Mul',[m,'half'],prefix+'y',shape)
    def norm(x,prefix):
        mean=add('ReduceMean',[x],prefix+'mean',[1,5,1],axes=[-1]);sub=add('Sub',[x,mean],prefix+'center',[1,5,4])
        sq=add('Pow',[sub,'two'],prefix+'square',[1,5,4]);var=add('ReduceMean',[sq],prefix+'var',[1,5,1],axes=[-1])
        eps=add('Add',[var,'epsilon'],prefix+'eps',[1,5,1]);std=add('Sqrt',[eps],prefix+'std',[1,5,1])
        div=add('Div',[sub,std],prefix+'unit',[1,5,4]);scale=add('Mul',[div,'lnw'],prefix+'scale',[1,5,4])
        return add('Add',[scale,'lnb'],prefix+'y',[1,5,4])
    c=add('Conv',['x','cw','cb'],'conv',[1,4,5],strides=[2],pads=[1,1],dilations=[1],group=1,kernel_shape=[3])
    c=gelu(c,[1,4,5],'g0');x=add('Transpose',[c],'sequence',[1,5,4],perm=[0,2,1]);ln=norm(x,'ln0')
    for n in ['qw','kw','vw','ow']:init(n,(rng.standard_normal((4,4))*.1).astype('f4'))
    q=add('MatMul',[ln,'qw'],'q',[1,5,4]);k=add('MatMul',[ln,'kw'],'k',[1,5,4]);v=add('MatMul',[ln,'vw'],'v',[1,5,4])
    k=add('Transpose',[k],'kt',[1,4,5],perm=[0,2,1]);scores=add('MatMul',[q,k],'scores',[1,5,5])
    probs=add('Softmax',[scores],'probabilities',[1,5,5],axis=-1);attention=add('MatMul',[probs,v],'attention',[1,5,4])
    attn=add('MatMul',[attention,'ow'],'out',[1,5,4]);res=add('Add',[x,attn],'res',[1,5,4]);ln=norm(res,'ln1')
    init('w1',(rng.standard_normal((4,8))*.1).astype('f4'));init('w2',(rng.standard_normal((8,4))*.1).astype('f4'))
    ff=add('MatMul',[ln,'w1'],'ff1',[1,5,8]);ff=gelu(ff,[1,5,8],'g1');ff=add('MatMul',[ff,'w2'],'ff2',[1,5,4])
    result=add('Add',[res,ff],'final',[1,5,4]);init('shape',np.array([0,-1,4],dtype='i8'));add('Reshape',[result,'shape'],'reshaped',[1,5,4])
    graph=h.make_graph(nodes,'small encoder',[h.make_tensor_value_info('x',T.FLOAT,[1,2,9])],[infos[-1]],inits)
    original=h.make_model(graph,opset_imports=[h.make_opsetid('',14)],ir_version=10)
    trace=copy.deepcopy(original);del trace.graph.output[:];trace.graph.output.extend(infos)
    return original,trace,(rng.standard_normal((1,2,9))*.25).astype('f4')

class EngineTests(unittest.TestCase):
    def test_all_operator_semantics_and_negative_axes(self):
        a=np.array([[1.,-2.,3.],[4.,5.,-6.]],dtype='f8');positive=np.abs(a)+.25
        cases=[('Add',[a,np.array([.5,1.,-2.])],{}),('Sub',[a,a/2],{}),('Mul',[a,a],{}),('Div',[a,positive],{}),
            ('Pow',[a,np.array(2.)],{}),('Sqrt',[positive],{}),('Erf',[a],{}),('ReduceMean',[a],dict(axes=[-1],keepdims=0)),
            ('Transpose',[a],dict(perm=[1,0])),('Reshape',[a,np.array([0,1,-1],dtype='i8')],{}),
            ('MatMul',[a,a.T],{}),('Softmax',[a],dict(axis=-1)),
            ('Conv',[a.reshape(1,2,3),np.array([[[1.,2.],[-3.,4.]]])],dict(strides=[2],pads=[1,0],dilations=[1],group=1))]
        self.assertEqual({op for op,x,attrs in cases},OPS)
        for op,values,attrs in cases:
            node=h.make_node(op,[f'x{i}' for i in range(len(values))],['y'],**attrs)
            result=execute(node,values)
            inputs=[h.make_tensor_value_info(f'x{i}',T.INT64 if v.dtype==np.int64 else T.DOUBLE,v.shape) for i,v in enumerate(values)]
            model=h.make_model(h.make_graph([node],'one',inputs,[h.make_tensor_value_info('y',T.DOUBLE,result.shape)]),opset_imports=[h.make_opsetid('',14)],ir_version=10)
            wanted=ReferenceEvaluator(model,new_ops=[Erf]).run(None,{f'x{i}':v for i,v in enumerate(values)})[0]
            np.testing.assert_allclose(result,wanted,rtol=1e-13,atol=1e-13,err_msg=op)

    def test_complete_small_network_and_promoted_ort(self):
        original,trace,features=fixture();saved={}
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);ledger=build(original,trace,folder,folder)
            promoted=onnx.load(folder/'model.onnx',load_external_data=False)
            by_name={i.name:i for i in promoted.graph.initializer}
            for item in original.graph.initializer:self.assertEqual(list(item.dims),list(by_name[item.name].dims))
            records=run(trace,folder,features.astype('f8'),lambda name,value:saved.update({name:value.copy()}))
            self.assertEqual(len(records),len(original.graph.node));self.assertEqual(len(ledger['lowered_convolutions']),1)
            stages=partition(onnx.load(folder/'model.onnx',load_external_data=False),folder);native={}
            native_run(folder,stages,'x',features.astype('f8'),lambda name,value:native.update({name:value.copy()}))
            direct=copy.deepcopy(trace)
            for i in direct.graph.initializer:
                value=tensor_array(i,folder)
                if value.dtype==np.float32:i.CopyFrom(nh.from_array(value.astype('f8'),i.name))
            for info in list(direct.graph.input)+list(direct.graph.output):info.type.tensor_type.elem_type=T.DOUBLE
            oracle=ReferenceEvaluator(direct,new_ops=[Erf]).run(None,{'x':features.astype('f8')})
            for info,expected in zip(trace.graph.output,oracle):
                np.testing.assert_allclose(saved[info.name],expected,rtol=1e-12,atol=1e-12,err_msg=info.name)
                np.testing.assert_allclose(native[info.name],expected,rtol=1e-12,atol=1e-12,err_msg=info.name)

    def test_changed_graph_or_unsupported_convolution_refused(self):
        original,trace,features=fixture();trace.graph.node[0].attribute[0].i=99
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):build(original,trace,Path(directory),Path(directory))
        node=h.make_node('Conv',['x','w'],['y'],group=2)
        with self.assertRaises(AssertionError):execute(node,[np.ones((1,2,5)),np.ones((2,2,3))])

    def test_double_erf_preserves_precision_and_signed_zero(self):
        from scipy.special import erf
        values=np.r_[np.linspace(-10,10,2001),np.array([-0.,0.,1e-20,-1e-20,1-1e-15,1+1e-15,8-1e-15,8+1e-15])]
        actual=double_erf(values);np.testing.assert_allclose(actual,erf(values),rtol=1e-15,atol=1e-15)
        self.assertTrue(np.signbit(actual[-8]));self.assertFalse(np.signbit(actual[-7]))

    def test_resource_refuses_missing_identity_limits_or_sampling(self):
        from audit import resources
        from common import LIMITS
        supervisor=dict(pid=1,birth=1);worker=dict(pid=2147483000,birth=2)
        run=dict(complete=True,code=0,seconds=2,started=4,ended=6,worker=worker,job=dict(id='fixture'),samples=2,
            preflight_available=LIMITS['preflight_available'],preflight_disk=LIMITS['disk'])
        row=dict(seconds=.5,**worker,rss=1000,available=LIMITS['available'],affinity=[2]);rows=[row,row|dict(seconds=1.5)]
        self.assertEqual(resources(run,rows,supervisor)['samples'],2)
        for key,value in [('birth',3),('pid',2147483001),('affinity',[0]),('rss',LIMITS['rss']),('available',0),('seconds',3)]:
            with self.assertRaises(AssertionError):resources(run,[rows[0],rows[1]|{key:value}],supervisor)
        for damaged in [run|dict(code=1),run|dict(complete=False),run|dict(samples=3),run|dict(seconds=LIMITS['seconds'])]:
            with self.assertRaises(AssertionError):resources(damaged,rows,supervisor)

if __name__=='__main__':unittest.main()
