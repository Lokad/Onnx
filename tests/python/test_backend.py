import os
from secrets import token_urlsafe
from typing import Dict

import math
import itertools
from unicodedata import decimal

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
    #np.testing.assert_almost_equal(output['Y'], exp_output)
    np.testing.assert_allclose(output['Y'], exp_output, rtol=1e-6, atol=1e-6)


    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    y = np.vectorize(math.erf)(x).astype(np.float32)
    output = backend.run_node(node_def, [x])
    np.testing.assert_allclose(output['Y'], y, rtol=1e-6, atol=1e-6)

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

def test_cast():
    test_cases = [
        (onnx.TensorProto.FLOAT),
        (onnx.TensorProto.UINT8), 
        (onnx.TensorProto.INT8),
        (onnx.TensorProto.UINT16),
        (onnx.TensorProto.INT16),
        (onnx.TensorProto.INT32),
        (onnx.TensorProto.INT64), 
        (onnx.TensorProto.BOOL),
        #(onnx.TensorProto.FLOAT16),
        (onnx.TensorProto.DOUBLE),
        #(onnx.TensorProto.COMPLEX64),
        #(onnx.TensorProto.COMPLEX128)
    ]
    for ty in test_cases:
      node_def = onnx.helper.make_node("Cast", ["input"], ["output"], to=ty)
      vector = np.array([2, 3])
      output = backend.run_node(node_def, [vector])
      np.testing.assert_equal(output["output"].dtype, onnx.helper.tensor_dtype_to_np_dtype(ty))

def test_concat():
    shape = [10, 20, 5]
    for axis in range(len(shape)):
        node_def = onnx.helper.make_node("Concat", ["X1", "X2"], ["Y"], axis=axis)
        x1 = _get_rnd_float32(shape=shape)
        x2 = _get_rnd_float32(shape=shape)
        output = backend.run_node(node_def, [x1, x2])
        np.testing.assert_almost_equal(output["Y"], np.concatenate((x1, x2), axis))

    test_cases = {
        "1d": ([1, 2], [3, 4]),
        "2d": ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        "3d": (
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ),
    }

    for _, values_ in test_cases.items():
        values = [np.asarray(v, dtype=np.float32) for v in values_]
        for axis in range(len(values[0].shape)):
            in_args = ["value" + str(k) for k in range(len(values))]
            node_def = onnx.helper.make_node(
                "Concat", inputs=list(in_args), outputs=["output"], axis=axis
            )
            output = backend.run_node(node_def, values)
            np.testing.assert_almost_equal(output["output"], np.concatenate(values, axis))

        for axis in range(-len(values[0].shape), 0):
            in_args = ["value" + str(k) for k in range(len(values))]
            node_def = onnx.helper.make_node(
                "Concat", inputs=list(in_args), outputs=["output"], axis=axis
            )
            output = backend.run_node(node_def, values)
            np.testing.assert_almost_equal(output["output"], np.concatenate(values, axis))

def _shape_reference_impl(x, start=None, end=None):  # type: ignore
    dims = x.shape[start:end]
    return np.array(dims).astype(np.int64)
    
def _test_shape(_, xval, start=None, end=None):  # type: ignore
    node_def = onnx.helper.make_node(
        "Shape", inputs=["x"], outputs=["y"], start=start, end=end
    )
    output = backend.run_node(node_def, [xval], start=start)
    np.testing.assert_almost_equal(output["y"], _shape_reference_impl(xval, start, end))
        

def test_shape():
    node_def = onnx.helper.make_node("Shape", ["X"], ["Y"])
    x = _get_rnd_float32(shape=[5, 10, 10, 3])
    output = backend.run_node(node_def, [x])
    np.testing.assert_allclose(output["Y"], np.shape(x))

    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    ).astype(np.float32)
    _test_shape("_example", x) 

    x = np.random.randn(3, 4, 5).astype(np.float32)

    _test_shape("", x)  # preserve names of original test cases

    _test_shape("_start_1", x, 1)

    _test_shape("_end_1", x, end=1)

    _test_shape("_start_negative_1", x, start=-1)

    _test_shape("_end_negative_1", x, end=-1)

    _test_shape("_start_1_end_negative_1", x, start=1, end=-1)

    _test_shape("_start_1_end_2", x, start=1, end=2)

    _test_shape("_clip_start", x, start=-10)

    _test_shape("_clip_end", x, end=10)


def test_gather():
    node_def = onnx.helper.make_node("Gather", ["X", "Y"], ["Z"])
    x = _get_rnd_float32(shape=[10, 10])
    y = np.array([[0, 1], [1, 2]])
    output = backend.run_node(node_def, [x, y])
    test_output = np.zeros((2, 2, 10))
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[0][i][j] = x[i][j]
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[1][i][j] = x[i + 1][j]
    np.testing.assert_almost_equal(output["Z"], test_output)
     # test negative indices
    y = np.array([[-10, -9], [1, -8]])
    output = backend.run_node(node_def, [x, y])
    np.testing.assert_almost_equal(output["Z"], test_output)
 
    node_def = onnx.helper.make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["y"],
            axis=0,
        )
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    y = np.take(data, indices, axis=0)

    output = backend.run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["y"],
            axis=1,
        )
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    y = np.take(data, indices, axis=1)
    output = backend.run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["y"], y)

    data = np.random.randn(3, 3).astype(np.float32)
    indices = np.array([[0, 2]])
    y = np.take(data, indices, axis=1)
    output = backend.run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["y"], y)
    node_def = onnx.helper.make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["y"],
            axis=0,
        )
    data = np.arange(10).astype(np.float32)
    indices = np.array([0, -9, -10])
    y = np.take(data, indices, axis=0)
    output = backend.run_node(node_def, [data, indices])
    np.testing.assert_almost_equal(output["y"], y)


def test_slice():
    # test case 1 with normal inputs
    axes = np.array([0, 1, 2])
    starts = np.array([0, 0, 0])
    ends = np.array([2, 2, 2])
    steps = np.array([1, 1, 1])
    node_def = onnx.helper.make_node("Slice",
                                  ["X", "starts", "ends", "axes", "steps"],
                                  ["S"])
    x = _get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
    output = backend.run_node(node_def, [x, starts, ends, axes, steps])
    np.testing.assert_almost_equal(output["S"], x[0:2, 0:2, 0:2])

    # test case 2 with negative, out-of-bound and default inputs
    axes = np.array([0, 2])
    starts = np.array([0, -7])
    ends = np.array([-8, 20])

    node_def = onnx.helper.make_node("Slice", ["X", "starts", "ends", "axes"],
                                  ["S"])
    x = _get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
    output = backend.run_node(node_def, [x, starts, ends, axes])
    np.testing.assert_almost_equal(output["S"], x[0:-8, :, -7:20])

    node_def = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes", "steps"],
            outputs=["y"],
        )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    y = x[0:3, 0:10]
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([3, 10], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)

    output = backend.run_node(node_def, [x, starts, ends, axes, steps])
    np.testing.assert_almost_equal(output["y"], y)

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([-1], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 0:-1]
    output = backend.run_node(node_def, [x, starts, ends, axes, steps])
    np.testing.assert_almost_equal(output["y"], y)


    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([1000], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 1000:1000]

    output = backend.run_node(node_def, [x, starts, ends, axes, steps])
    np.testing.assert_almost_equal(output["y"], y)

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([1], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 1:1000]

    output = backend.run_node(node_def, [x, starts, ends, axes, steps])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends"],
            outputs=["y"],
    )
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    y = x[:, :, 3:4]

    output = backend.run_node(node_def, [x, starts, ends])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends", "axes"],
        outputs=["y"],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    y = x[:, :, 3:4]

    output = backend.run_node(node_def, [x, starts, ends, axes])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
            "Slice",
            inputs=["x", "starts", "ends", "axes", "steps"],
            outputs=["y"],
        )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([20, 10, 4], dtype=np.int64)
    ends = np.array([0, 0, 1], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    steps = np.array([-1, -3, -2]).astype(np.int64)
    y = x[20:0:-1, 10:0:-3, 4:1:-2]
    output = backend.run_node(node_def, [x, starts, ends, axes, steps])
    np.testing.assert_almost_equal(output["y"], y)

def test_unsqueeze():
    x = np.random.randn(3, 4, 5).astype(np.float32)

    for i in range(x.ndim):
        axes = np.array([i]).astype(np.int64)
        node_def = onnx.helper.make_node(
            "Unsqueeze",
            inputs=["x", "axes"],
            outputs=["y"],
        )
        y = np.expand_dims(x, axis=i)

        output = backend.run_node(node_def, [x, axes])
        np.testing.assert_almost_equal(output["y"], y)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([1, 4]).astype(np.int64)

    node_def = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    y = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=4)
    output = backend.run_node(node_def, [x, axes])
    np.testing.assert_almost_equal(output["y"], y)

        
    axes = np.array([2, 4, 5]).astype(np.int64)
    node_def = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)
    output = backend.run_node(node_def, [x, axes])
    np.testing.assert_almost_equal(output["y"], y)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([5, 4, 2]).astype(np.int64)

    node_def = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)
    output = backend.run_node(node_def, [x, axes])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2]).astype(np.int64)
    y = np.expand_dims(x, axis=-2)
    output = backend.run_node(node_def, [x, axes])
    np.testing.assert_almost_equal(output["y"], y)


def test_reduce_sum():
    x = _get_rnd_float32(shape=[5, 10, 10, 3]).astype(np.int32)
    axes = np.array([1, 2], dtype=np.int64)
    node_def = onnx.helper.make_node("ReduceSum", ["data", "axes"], ["Y"])
    output = backend.run_node(node_def, [x, axes])
    np.testing.assert_allclose(output["Y"],
                                np.sum(x, (1, 2), keepdims=True),
                                rtol=1e-3)
    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = 0

    node_def = onnx.helper.make_node(
        "ReduceSum", inputs=["data", "axes"], outputs=["reduced"]
    )

    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int32
    )
    reduced = np.sum(data, axis=1, keepdims=True)

    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

    node_def = onnx.helper.make_node(
        "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=0
    )
    reduced = np.sum(data, axis=1, keepdims=False)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

    node_def = onnx.helper.make_node(
        "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=1
    )
    data = np.array([[ 0.0569, -0.2475,  0.0737, -0.3429],
        [-0.2993,  0.9138,  0.9337, -1.6864],
        [ 0.1132,  0.7892, -0.1003,  0.5688],
        [ 0.3637, -0.9906, -0.4752, -1.5197]]
    )
    axes = np.array([1], dtype=np.int64)
    reduced = np.sum(data, axis=1, keepdims=True)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

    data = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    axes = np.array([2, 1], dtype=np.int64)
    reduced = np.sum(data, axis=(2,1), keepdims=True)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

    
def test_reduce_mean():
    node_def = onnx.helper.make_node("ReduceMean", ["X", "axes"], ["y"])
    x = np.array([[1., 1.], [2., 2.]])
    axes = np.array([], dtype=np.int64)
    output = backend.run_node(node_def, [x, axes])
    np.testing.assert_almost_equal(output["y"], np.mean(x, keepdims=True))
    axes = np.array([0], dtype=np.int64)
    output = backend.run_node(node_def, [x, axes])
    np.testing.assert_almost_equal(output["y"],
                                np.mean(x, (0), keepdims=True))
    axes = np.array([1], dtype=np.int64)
    output = backend.run_node(node_def, [x, axes])
    np.testing.assert_almost_equal(output["y"],
                                np.mean(x, (1), keepdims=True))

    node_def = onnx.helper.make_node("ReduceMean", ["X", "axes"], ["y"])
    x = _get_rnd_float32(shape=[5, 10, 10, 3])
    output = backend.run_node(node_def, [x, np.array([1, 2])])
    np.testing.assert_almost_equal(output["y"],
                               np.mean(x, (1, 2), keepdims=True))
    
    axes = np.array([1], dtype=np.int64)
    keepdims = 0

    node_def = onnx.helper.make_node(
        "ReduceMean",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )

    data = np.array(
        [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
        dtype=np.float32,
    )
    reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

    np.random.seed(0)
    shape = [3, 2, 2]
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)


    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = 1

    node_def = onnx.helper.make_node(
            "ReduceMean",
            inputs=["data", "axes"],
            outputs=["reduced"],
            keepdims=keepdims,
        )

    data = np.array(
        [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
        dtype=np.float32,
    )
    reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

    shape = [3, 2, 2]
    axes = np.array([], dtype=np.int64)
    keepdims = 1

    node_def = onnx.helper.make_node(
        "ReduceMean",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )

    data = np.array(
        [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
        dtype=np.float32,
    )
    reduced = np.mean(data, axis=None, keepdims=True)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = 1

    node_def = onnx.helper.make_node(
        "ReduceMean",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )

    data = np.array(
        [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
        dtype=np.float32,
    )
    reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
    output = backend.run_node(node_def, [data, axes])
    np.testing.assert_almost_equal(output["reduced"], reduced)

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / s

def test_softmax():
    node_def = onnx.helper.make_node("Softmax", ["X"], ["Y"], axis=0)
    x = _get_rnd_float32(shape=[3, 4, 5])
    output = backend.run_node(node_def, [x])
    x_max = np.max(x, axis=0, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=0, keepdims=True)
    y = tmp / s
    np.testing.assert_almost_equal(output["Y"], y)

    x = np.array([-1, 0., 1.])
    output = backend.run_node(node_def, [x])
    x_max = np.max(x, axis=0, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=0, keepdims=True)
    y = tmp / s
    np.testing.assert_almost_equal(output["Y"], y)

    node_def = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
        )
    x = np.array([[-1, 0, 1]]).astype(np.float32)
    # expected output [[0.09003058, 0.24472848, 0.66524094]]
    y = _softmax(x, axis=1)
    output = backend.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["y"], y)

    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
        # expected output
        # [[0.032058604 0.08714432  0.23688284  0.6439143  ]
        # [0.032058604 0.08714432  0.23688284  0.6439143  ]]
    y = _softmax(x)
    output = backend.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
            axis = 0
        )
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    y = _softmax(x, axis=0)
    output = backend.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
            axis=1,
        )
    y = _softmax(x, axis=1)
    output = backend.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
            axis=2,
        )
    y = _softmax(x, axis=2)
    output = backend.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
            axis=-1,
        )
    y = _softmax(x, axis=-1)
    output = backend.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["y"], y)

    node_def = onnx.helper.make_node(
            "Softmax",
            inputs=["x"],
            outputs=["y"],
        )
    output = backend.run_node(node_def, [x])
    np.testing.assert_almost_equal(output["y"], y, decimal=6)