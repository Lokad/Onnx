import numpy as np
from onnx import helper, TensorProto

rng = np.random.default_rng(20260912)

def save(path, node, inp, out, inits):
    g = helper.make_graph([node], "oneop", inp, out, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    import os
    os.makedirs(path, exist_ok=True)
    import onnx
    onnx.save(m, os.path.join(path, "model.onnx"))
    print("wrote", os.path.join(path, "model.onnx"))

def finit(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, list(arr.shape), arr.reshape(-1).astype(np.float32))

def tinfo(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))

# 1. top 1x1 conv GEMM tile: 1024x256 @ 256x196
a_shape, b_shape, c_shape = [1024, 256], [256, 196], [196]
b = (rng.random(b_shape) * 2 - 1).astype(np.float32)
c = np.zeros(c_shape, dtype=np.float32)
save("tests/Lokad.Onnx.Bench/oneop/gemm_1024x256",
     helper.make_node("Gemm", ["a", "b", "c"], ["y"], alpha=1.0, beta=1.0),
     [tinfo("a", a_shape)], [tinfo("y", [1024, 196])],
     [finit("b", b), finit("c", c)])

# 2. top 3x3 conv GEMM tile: 512x4608 @ 4608x196
a_shape, b_shape = [512, 4608], [4608, 196]
# K=4608 sums would drown the 1e-4 one-op gate in fp-ordering noise; scale B
# to +-0.125 (timing-neutral, same draws) so agreement stays meaningful.
b = ((rng.random(b_shape) * 2 - 1) * 0.125).astype(np.float32)
c = np.zeros([196], dtype=np.float32)
save("tests/Lokad.Onnx.Bench/oneop/gemm_512x4608",
     helper.make_node("Gemm", ["a", "b", "c"], ["y"], alpha=1.0, beta=1.0),
     [tinfo("a", a_shape)], [tinfo("y", [512, 196])],
     [finit("b", b), finit("c", c)])

# 3. full 3x3 conv 512ch @14x14, no bias (mirrors conv_3x3 case)
w = (rng.random([512, 512, 3, 3]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_3x3_512",
     helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
     [tinfo("x", [1, 512, 14, 14])], [tinfo("y", [1, 512, 14, 14])],
     [finit("w", w)])

# 4. full 1x1 conv 256->1024 @14x14, no bias
w = (rng.random([1024, 256, 1, 1]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_1x1_1024",
     helper.make_node("Conv", ["x", "w"], ["y"]),
     [tinfo("x", [1, 256, 14, 14])], [tinfo("y", [1, 1024, 14, 14])],
     [finit("w", w)])