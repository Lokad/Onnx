"""Check graph dependency closure and suffix equivalence on explicit examples."""
import unittest
from protocol import *
import onnx
from onnx import helper, numpy_helper, TensorProto


def model(extra_input=False):
    a = np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    b = np.array([[.5,-2.],[1.,.25]],dtype=np.float32)
    initializers = [numpy_helper.from_array(a,'a'),numpy_helper.from_array(b,'b')]
    nodes = [helper.make_node('Add',['x','a'],['cut']),helper.make_node('MatMul',['cut','b'],['product']),
             helper.make_node('Add',['product','x' if extra_input else 'a'],['y'])]
    return helper.make_model(helper.make_graph(nodes,'suffix',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[2,2])],
                            [helper.make_tensor_value_info('y',TensorProto.FLOAT,[2,2])],initializers))


class SuffixTests(unittest.TestCase):
    def test_full_and_cut_exact_for_both_modes(self):
        graph = model();x=np.array([[3.,-4.],[5.,2.]],dtype=np.float32)
        initial = {i.name:numpy_helper.to_array(i) for i in graph.graph.initializer}
        for mode in MODES:
            values = dict(initial,x=x); full = {}
            for node in graph.graph.node:
                values[node.output[0]] = calculate(node,[values[n] for n in node.input],mode)
                full[node.output[0]] = values[node.output[0]].copy()
            cut = full['cut'].copy();before=raw(cut);got={}
            records,dots=run_suffix(graph,'cut',cut,mode,lambda n,v:got.update({n:v.copy()}))
            self.assertEqual([r['index'] for r in records],[1,2])
            self.assertEqual(raw(cut),before)
            for name in ['product','y']:
                np.testing.assert_array_equal(got[name].view(np.uint32),full[name].view(np.uint32))
            self.assertEqual(len(dots),1 if mode=='wide-matmul' else 0)

    def test_extra_dependency_missing_cut_and_unused_cut_refuse(self):
        with self.assertRaisesRegex(AssertionError,'Unbound'):
            select_suffix(model(True),'cut')
        with self.assertRaisesRegex(AssertionError,'exactly one'):
            select_suffix(model(),'missing')
        graph=model();graph.graph.node[1].input[0]='a'
        with self.assertRaisesRegex(AssertionError,'does not consume'):
            select_suffix(graph,'cut')

    def test_original_node_bytes_and_rounding_contract(self):
        graph=model();start,nodes=select_suffix(graph,'cut')
        self.assertEqual([n.SerializeToString() for n in nodes],[n.SerializeToString() for n in graph.graph.node[start:]])
        x=np.zeros((1,1500,1280),dtype=np.float64);x.flat[0]=1.+2**-24;x.flat[1]=1.+3*2**-24
        y=narrow_stem(x)
        self.assertEqual(y.dtype,np.float32)
        self.assertEqual(float(y.flat[0]),1.)
        self.assertEqual(float(y.flat[1]),1.+2**-22)
        with self.assertRaises(AssertionError):narrow_stem(x[:,:,0])
        x.flat[2]=float('nan')
        with self.assertRaises(AssertionError):narrow_stem(x)


if __name__=='__main__':unittest.main()
