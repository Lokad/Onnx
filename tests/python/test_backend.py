import os
import onnx
import numpy as np
from typing import Dict
from onnx.reference import ReferenceEvaluator

from interop import backend
from interop.tensors import ndarray_eq

file_dir = os.path.dirname(os.path.realpath(__file__))

mnist4 = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "images", "mnist4.png") + "::mnist"
onnx_model_file = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx")

backend.set_debug_mode()

def reference_eval_node(node:onnx.NodeProto, args:Dict[str, np.ndarray]):
    sess = ReferenceEvaluator(node)
    return sess.run(None, args)

def test_load_graph():
    g = backend.load_graph(os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx"))
    assert g.Nodes.Count == 12

def test_model_run():
    node = onnx.helper.make_node(
            name="Add1",
            op_type="Add",
            inputs=["x", "y"],
            outputs=["Add1"],
    )
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT32, [4,5])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT32, [4,5])
    Add1 = onnx.helper.make_tensor_value_info("Add1", onnx.TensorProto.INT32, [4,5])
    graph = onnx.helper.make_graph([node], "graph1", [x, y], [Add1])
    model = onnx.helper.make_model(graph)
    rep = backend.prepare(model)
    assert rep.graph.Nodes.Count == 1
    p = np.arange(0, 20).reshape(4, 5)
    r = rep.run([p, p])
    a = r[0]
    assert len(a.shape) == 2
    assert a.shape[1] == 5
    assert a[0,1] == 2  
    
def test_model_file_run():
    rep = backend.prepare_file(onnx_model_file)
    assert rep.graph.Nodes.Count == 12
    inputs = [mnist4]
    file_args = [0]
    r = rep.run(inputs, file_args=file_args, use_initializers=True)

def test_node_run():
    node = onnx.helper.make_node(
            name="Add2",
            op_type="Add",
            inputs=["x", "y"],
            outputs=["Add2"],
    )
    x = np.arange(0, 20).reshape(4, 5)
    y = np.arange(0, 20).reshape(4, 5)
    a = backend.run_node(node, [x, y])
    assert len(a[0].shape) == 2
    assert a[0].shape[1] == 5
    assert a[0][0,2] == 4  

def test_model_node_run():
    inp1 = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "images", "mnist4.png") + "::1:10"
    inputs = [inp1]
    file_args = [0]
    rep = backend.prepare_file(onnx_model_file)
    r = rep.run_node('Plus214', inputs, file_args=file_args, use_initializers=True)
    assert set(r[0].shape) == {1, 10}

def test_mnist_model_run():
    an = np.ones((1,1,28,28), np.float32)
    rep = backend.prepare_file(onnx_model_file)
    
    node = rep.get_onnx_node('Times212_reshape1')
    r = rep.run_node(node, [], use_initializers = True)
    ref_r = reference_eval_node(node, {'Parameter193': rep.get_initializer('Parameter193'), 'Parameter193_reshape1_shape':rep.get_initializer('Parameter193_reshape1_shape')})
    assert ndarray_eq(ref_r[0], r[0])

    node = rep.get_onnx_node('Plus30')
    r = rep.run_node(node, [mnist4], file_args=[0], use_initializers=True)
    ref_r = reference_eval_node(node, {'Convolution28_Output_0': rep.get_input_ndarray_from_file_arg(mnist4), 'Parameter6':rep.get_initializer('Parameter6')})
    np.testing.assert_almost_equal(ref_r[0], r[0])

    #node = rep.get_onnx_node('Convolution28')
    #r = rep.run_node(node, [mnist4, np.ones((1,1,28,28), np.float3)], file_args=[0])
    #np.testing.assert_almost_equal(ref_r[0], r[0])

   
    