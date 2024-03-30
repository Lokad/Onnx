import os
from typing import Dict

import math
import itertools

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from interop import backend

file_dir = os.path.dirname(os.path.realpath(__file__))

mnist4 = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "images", "mnist4.png") + "::mnist"
onnx_model_file = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx")

backend.set_debug_mode()

def _get_rnd_float32(low=-1.0, high=1.0, shape=None):
    output = np.random.uniform(low, high, shape)
    if shape is None:
      return np.float32(output)
    else:
      return output.astype(np.float32)

def _get_rnd_float16(low=-1.0, high=1.0, shape=None):
    output = np.random.uniform(low, high, shape)
    if shape is None:
      return np.float16(output)
    else:
      return output.astype(np.float16)

def _get_rnd_int(low, high=None, shape=None, dtype=np.int32):
    return np.random.randint(low, high, size=shape, dtype=dtype)

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
    rep = backend.prepare_file(onnx_model_file)
    
    node = rep.get_onnx_node('Times212_reshape1')
    r = rep.run_node(node, [], use_initializers = True)
    ref_r = reference_eval_node(node, {'Parameter193': rep.get_initializer('Parameter193'), 'Parameter193_reshape1_shape':rep.get_initializer('Parameter193_reshape1_shape')})
    np.testing.assert_equal(ref_r[0], r[0])

    node = rep.get_onnx_node('Plus30')
    r = rep.run_node(node, [mnist4], file_args=[0], use_initializers=True)
    ref_r = reference_eval_node(node, {'Convolution28_Output_0': rep.get_input_ndarray_from_file_arg(mnist4), 'Parameter6':rep.get_initializer('Parameter6')})
    np.testing.assert_almost_equal(ref_r[0], r[0])

    i = ref_r[0]
    node = rep.get_onnx_node('Pooling66')
    r = rep.run_node(node, [i])
    ref_r = reference_eval_node(node, {'ReLU32_Output_0': i})
    np.testing.assert_almost_equal(ref_r[0], r[0], 4)


    node = rep.get_onnx_node('ReLU32')
    r = rep.run_node(node, [mnist4], file_args=[0])
    ref_r = reference_eval_node(node, {'Plus30_Output_0': rep.get_input_ndarray_from_file_arg(mnist4)})
    np.testing.assert_almost_equal(ref_r[0], r[0])
    
    node = rep.get_onnx_node('Convolution28')
    r = rep.run_node(node, [mnist4], file_args=[0], use_initializers=True)
    ref_r = reference_eval_node(node, {'Input3': rep.get_input_ndarray_from_file_arg(mnist4), 'Parameter5': rep.get_initializer('Parameter5')})
    np.testing.assert_almost_equal(ref_r[0], r[0], decimal=4)

def test_add():
    node_def = onnx.helper.make_node("Add", ["X", "Y"], ["Z"], "Add")
    x = _get_rnd_float32(shape=[5, 10, 5, 5])
    y = _get_rnd_float32(shape=[10, 1, 1])
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.add(x, y.reshape([1, 10, 1, 1])))
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                   np.add(x, y))

def test_sub():
    node_def = onnx.helper.make_node("Sub", ["X", "Y"], ["Z"], "Sub1")
    x = _get_rnd_float32(shape=[10, 10])
    y = _get_rnd_float32(shape=[10, 10])
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.subtract(x, y))

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([3, 2, 1]).astype(np.float32)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x - y)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x - y)

    x = np.random.randint(12, 24, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(12, size=(3, 4, 5), dtype=np.uint8)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x - y)


def test_mul():
    node_def = onnx.helper.make_node("Mul", ["X", "Y"], ["Z"], "Mul1")
    x = _get_rnd_float32(shape=[5, 10, 5, 5])
    y = _get_rnd_float32(shape=[10, 1, 1])
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"],
                                    np.multiply(x, y.reshape([1, 10, 1, 1])))
    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.float32)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x * y)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x * y)

    x = np.random.randint(4, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x * y)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x * y)

def test_div():
    node_def = onnx.helper.make_node("Div", ["X", "Y"], ["Z"], "Div1")
    x = _get_rnd_float32(shape=[10, 10])
    y = _get_rnd_float32(shape=[10, 10])
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], np.divide(x, y))

    x = np.array([3, 4]).astype(np.float32)
    y = np.array([1, 2]).astype(np.float32)
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x / y)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x / y)

    x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8) + 1
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x // y)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32) + 1.0
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], x / y)

def test_erf():
    node_def = onnx.helper.make_node("Erf", ["X"], ["Y"], "Erf1")
    x = _get_rnd_float32(shape=[3, 4, 5])
    output = backend.run_node(node_def, [x])
    exp_output = np.vectorize(math.erf)(x).astype(np.float32)
    np.testing.assert_allclose(output['Y'], exp_output, rtol=1e-6, atol=1e-6)

def test_transpose():
    node_def = onnx.helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1], name="Transpose")
    x = _get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
    output = backend.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.transpose(x, (0, 2, 1)))

    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    node_def = onnx.helper.make_node(
        "Transpose", inputs=["data"], outputs=["transposed"], name="Transpose"
    )
    transposed = np.transpose(data)
    output = backend.run_node(node_def, [data])
    np.testing.assert_almost_equal(output["transposed"], transposed)

    permutations = list(itertools.permutations(np.arange(len(shape))))
    for _, permutation in enumerate(permutations):
        node_def = onnx.helper.make_node(
            "Transpose",
            inputs=["data"],
            outputs=["transposed"],
            perm=permutation,
            name="Transposed"
        )
        output = backend.run_node(node_def, [data])
        transposed = np.transpose(data, permutation)
        np.testing.assert_almost_equal(output["transposed"], transposed)

def test_constant():
    shape = [16, 16]
    values = np.random.randn(*shape).flatten().astype(float)
    const2_onnx = onnx.helper.make_tensor("const2", onnx.TensorProto.DOUBLE, shape,
                                     values)
    node_def = onnx.helper.make_node("Constant", [], ["Y"], value=const2_onnx)
    output = backend.run_node(node_def, [])
    np.testing.assert_equal(output["Y"].shape, shape)
    np.testing.assert_almost_equal(output["Y"].flatten(), values)

    values = np.random.randn(5, 5).astype(np.float32)
    node_def = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["values"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.FLOAT,
            dims=values.shape,
            vals=values.flatten().astype(float),
        ),
    )
    output = backend.run_node(node_def, [])
    np.testing.assert_almost_equal(output["values"], values)

    