import numpy as np
from onnx import helper, TensorProto

rng = np.random.default_rng(20260912)

def save(path, node, inp, out, inits, opset=14):
    g = helper.make_graph([node], "oneop", inp, out, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
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
# 5. e5 MLP up tile (batched 3D): 1x30x384 @ 384x1536
b = (rng.random([384, 1536]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_30x384x1536",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 30, 384])], [tinfo("y", [1, 30, 1536])],
     [finit("b", b)])

# 6. e5 MLP down tile (batched 3D): 1x30x1536 @ 1536x384
b = (rng.random([1536, 384]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_30x1536x384",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 30, 1536])], [tinfo("y", [1, 30, 384])],
     [finit("b", b)])

# 7. e5 QKV tile (batched 3D): 1x30x384 @ 384x384
b = (rng.random([384, 384]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_30x384x384",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 30, 384])], [tinfo("y", [1, 30, 384])],
     [finit("b", b)])

# 8. e5 intermediate Gelu: 1x30x1536
save("tests/Lokad.Onnx.Bench/oneop/gelu_30x1536",
     helper.make_node("Gelu", ["x"], ["y"]),
     [tinfo("x", [1, 30, 1536])], [tinfo("y", [1, 30, 1536])],
     [], opset=20)

# 9. e5 QKV shuffle transpose: 1x30x12x32 perm 0,2,1,3
save("tests/Lokad.Onnx.Bench/oneop/transpose_30x12x32",
     helper.make_node("Transpose", ["x"], ["y"], perm=[0, 2, 1, 3]),
     [tinfo("x", [1, 30, 12, 32])], [tinfo("y", [1, 12, 30, 32])],
     [])

# 10. GPT-2 c_attn tile: 4x768 @ 768x2304 + bias
b = (rng.random([768, 2304]) * 2 - 1).astype(np.float32)
c = (rng.random([2304]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/gemm_4x768x2304",
     helper.make_node("Gemm", ["a", "b", "c"], ["y"], alpha=1.0, beta=1.0),
     [tinfo("a", [4, 768])], [tinfo("y", [4, 2304])],
     [finit("b", b), finit("c", c)])

# 11. GPT-2 c_proj tile: 4x3072 @ 3072x768 + bias
b = (rng.random([3072, 768]) * 2 - 1).astype(np.float32)
c = (rng.random([768]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/gemm_4x3072x768",
     helper.make_node("Gemm", ["a", "b", "c"], ["y"], alpha=1.0, beta=1.0),
     [tinfo("a", [4, 3072])], [tinfo("y", [4, 768])],
     [finit("b", b), finit("c", c)])

# 12. DINOv3 MLP up tile (batched 3D): 1x201x384 @ 384x1536
b = (rng.random([384, 1536]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x201x384x1536",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 201, 384])], [tinfo("y", [1, 201, 1536])],
     [finit("b", b)])

# 13. DINOv3 MLP down tile (batched 3D): 1x201x1536 @ 1536x384
b = (rng.random([1536, 384]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x201x1536x384",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 201, 1536])], [tinfo("y", [1, 201, 384])],
     [finit("b", b)])

# 14. DINOv3 QKV tile (batched 3D): 1x201x384 @ 384x384
b = (rng.random([384, 384]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x201x384x384",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 201, 384])], [tinfo("y", [1, 201, 384])],
     [finit("b", b)])

# 15. DINOv3 attention softmax: 6x201x201
save("tests/Lokad.Onnx.Bench/oneop/softmax_6x201x201",
     helper.make_node("Softmax", ["x"], ["y"], axis=-1),
     [tinfo("x", [6, 201, 201])], [tinfo("y", [6, 201, 201])],
     [])

# 16. DINOv3 intermediate Gelu: 1x201x1536
save("tests/Lokad.Onnx.Bench/oneop/gelu_1x201x1536",
     helper.make_node("Gelu", ["x"], ["y"]),
     [tinfo("x", [1, 201, 1536])], [tinfo("y", [1, 201, 1536])],
     [], opset=20)

# 17. DINOv3 QKV shuffle transpose: 1x201x6x64 perm 0,2,1,3
save("tests/Lokad.Onnx.Bench/oneop/transpose_1x201x6x64",
     helper.make_node("Transpose", ["x"], ["y"], perm=[0, 2, 1, 3]),
     [tinfo("x", [1, 201, 6, 64])], [tinfo("y", [1, 6, 201, 64])],
     [])

# 18. e5 128-token MLP up tile (batched 3D): 1x128x384 @ 384x1536
b = (rng.random([384, 1536]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x128x384x1536",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 128, 384])], [tinfo("y", [1, 128, 1536])],
     [finit("b", b)])

# 19. e5 128-token MLP down tile (batched 3D): 1x128x1536 @ 1536x384
b = (rng.random([1536, 384]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x128x1536x384",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 128, 1536])], [tinfo("y", [1, 128, 384])],
     [finit("b", b)])

# 20. e5 512-token MLP up tile (batched 3D): 1x512x384 @ 384x1536
b = (rng.random([384, 1536]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x512x384x1536",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 512, 384])], [tinfo("y", [1, 512, 1536])],
     [finit("b", b)])

# 21. e5 512-token MLP down tile (batched 3D): 1x512x1536 @ 1536x384
b = (rng.random([1536, 384]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x512x1536x384",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 512, 1536])], [tinfo("y", [1, 512, 384])],
     [finit("b", b)])

# 22. DINOv3 attention scores (batched): 1x6x201x64 @ 1x6x64x201
b = (rng.random([64, 201]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/attn_1x6x201x64",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 6, 201, 64])], [tinfo("y", [1, 6, 201, 201])],
     [finit("b", b)])

# 23. DINOv3 attention context (batched): 1x6x201x201 @ 1x6x201x64
b = (rng.random([201, 64]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/attn_1x6x201x201",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 6, 201, 201])], [tinfo("y", [1, 6, 201, 64])],
     [finit("b", b)])

# 24. e5 attention scores (batched): 1x12x30x32 @ 32x30
b = (rng.random([32, 30]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/attn_1x12x30x32",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 12, 30, 32])], [tinfo("y", [1, 12, 30, 30])],
     [finit("b", b)])

# 25. e5 attention context (batched): 1x12x30x30 @ 30x32
b = (rng.random([30, 32]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/attn_1x12x30x30",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 12, 30, 30])], [tinfo("y", [1, 12, 30, 32])],
     [finit("b", b)])

# 26. e5 8-token MLP up tile (batched 3D): 1x8x384 @ 384x1536
b = (rng.random([384, 1536]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x8x384x1536",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 8, 384])], [tinfo("y", [1, 8, 1536])],
     [finit("b", b)])

# 27. e5 8-token MLP down tile (batched 3D): 1x8x1536 @ 1536x384
b = (rng.random([1536, 384]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/matmul_1x8x1536x384",
     helper.make_node("MatMul", ["a", "b"], ["y"]),
     [tinfo("a", [1, 8, 1536])], [tinfo("y", [1, 8, 384])],
     [finit("b", b)])
# 28. e5 attention block (multi-op region fixture, 12 heads x 30 tok, d=384/32):
# Q/K/V projections, head-layout transposes, scaled causal-free scores, softmax,
# context, merge transpose, output projection. Seeded weights; the runner feeds a
# runtime-shaped activation. Exercises the dispatch/transpose/copy traffic that
# single-op tiles cannot show (A01/M03 material).
def save_graph(path, nodes, inp, out, inits, opset=14):
    g = helper.make_graph(nodes, "multiop", inp, out, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
    import os
    os.makedirs(path, exist_ok=True)
    import onnx
    onnx.save(m, os.path.join(path, "model.onnx"))
    print("wrote", os.path.join(path, "model.onnx"))

wq = (rng.random([384, 384]) * 2 - 1).astype(np.float32)
wk = (rng.random([384, 384]) * 2 - 1).astype(np.float32)
wv = (rng.random([384, 384]) * 2 - 1).astype(np.float32)
wo = (rng.random([384, 384]) * 2 - 1).astype(np.float32)
scale = np.array(1.0 / np.sqrt(32.0), dtype=np.float32)
attn_nodes = [
    helper.make_node("MatMul", ["x", "wq"], ["q3"], name="qproj"),
    helper.make_node("MatMul", ["x", "wk"], ["k3"], name="kproj"),
    helper.make_node("MatMul", ["x", "wv"], ["v3"], name="vproj"),
    helper.make_node("Reshape", ["q3", "qshape"], ["q4"], name="qreshape"),
    helper.make_node("Reshape", ["k3", "kshape"], ["k4"], name="kreshape"),
    helper.make_node("Reshape", ["v3", "vshape"], ["v4"], name="vreshape"),
    helper.make_node("Transpose", ["q4"], ["q"], perm=[0, 2, 1, 3], name="qtranspose"),
    helper.make_node("Transpose", ["k4"], ["kt"], perm=[0, 2, 3, 1], name="ktranspose"),
    helper.make_node("Transpose", ["v4"], ["v"], perm=[0, 2, 1, 3], name="vtranspose"),
    helper.make_node("MatMul", ["q", "kt"], ["scores"], name="scores"),
    helper.make_node("Div", ["scores", "scale"], ["scaled"], name="scalediv"),
    helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1, name="softmax"),
    helper.make_node("MatMul", ["probs", "v"], ["ctx4"], name="context"),
    helper.make_node("Transpose", ["ctx4"], ["ctx3"], perm=[0, 2, 1, 3], name="ctxmerge"),
    helper.make_node("Reshape", ["ctx3", "yshape"], ["merged"], name="mergereshape"),
    helper.make_node("MatMul", ["merged", "wo"], ["y"], name="outproj"),
]
shape4132 = np.array([1, 30, 12, 32], dtype=np.int64)
mergeshape = np.array([1, 30, 384], dtype=np.int64)
save_graph("tests/Lokad.Onnx.Bench/oneop/attnblock_e5_30", attn_nodes,
     [tinfo("x", [1, 30, 384])], [tinfo("y", [1, 30, 384])],
     [finit("wq", wq), finit("wk", wk), finit("wv", wv), finit("wo", wo),
      finit("scale", scale),
      helper.make_tensor("qshape", TensorProto.INT64, [4], shape4132),
      helper.make_tensor("kshape", TensorProto.INT64, [4], shape4132),
      helper.make_tensor("vshape", TensorProto.INT64, [4], shape4132),
      helper.make_tensor("yshape", TensorProto.INT64, [3], mergeshape)])

# 28b. e5 8-token attention block (A01 8tok picture): same region as #28 at S=8.
shape8132 = np.array([1, 8, 12, 32], dtype=np.int64)
mergeshape8 = np.array([1, 8, 384], dtype=np.int64)
attn8_nodes = [
    helper.make_node("MatMul", ["x", "wq"], ["q3"], name="qproj"),
    helper.make_node("MatMul", ["x", "wk"], ["k3"], name="kproj"),
    helper.make_node("MatMul", ["x", "wv"], ["v3"], name="vproj"),
    helper.make_node("Reshape", ["q3", "qshape8"], ["q4"], name="qreshape"),
    helper.make_node("Reshape", ["k3", "kshape8"], ["k4"], name="kreshape"),
    helper.make_node("Reshape", ["v3", "vshape8"], ["v4"], name="vreshape"),
    helper.make_node("Transpose", ["q4"], ["q"], perm=[0, 2, 1, 3], name="qtranspose"),
    helper.make_node("Transpose", ["k4"], ["kt"], perm=[0, 2, 3, 1], name="ktranspose"),
    helper.make_node("Transpose", ["v4"], ["v"], perm=[0, 2, 1, 3], name="vtranspose"),
    helper.make_node("MatMul", ["q", "kt"], ["scores"], name="scores"),
    helper.make_node("Div", ["scores", "scale"], ["scaled"], name="scalediv"),
    helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1, name="softmax"),
    helper.make_node("MatMul", ["probs", "v"], ["ctx4"], name="context"),
    helper.make_node("Transpose", ["ctx4"], ["ctx3"], perm=[0, 2, 1, 3], name="ctxmerge"),
    helper.make_node("Reshape", ["ctx3", "yshape8"], ["merged"], name="mergereshape"),
    helper.make_node("MatMul", ["merged", "wo"], ["y"], name="outproj"),
]
save_graph("tests/Lokad.Onnx.Bench/oneop/attnblock_e5_8", attn8_nodes,
     [tinfo("x", [1, 8, 384])], [tinfo("y", [1, 8, 384])],
     [finit("wq", wq), finit("wk", wk), finit("wv", wv), finit("wo", wo),
      finit("scale", scale),
      helper.make_tensor("qshape8", TensorProto.INT64, [4], shape8132),
      helper.make_tensor("kshape8", TensorProto.INT64, [4], shape8132),
      helper.make_tensor("vshape8", TensorProto.INT64, [4], shape8132),
      helper.make_tensor("yshape8", TensorProto.INT64, [3], mergeshape8)])

# 29. resnet bottleneck (multi-op region fixture, 128ch @28x28, inner 32):
# pointwise -> 3x3 -> pointwise with residual add and relu epilogues. Downscaled
# from layer2.1 (512ch/inner-128) for fixture iteration speed; the region pattern
# (layout transitions, fused output work) is what C02/C03 prototypes measure.
w1 = (rng.random([32, 128, 1, 1]) * 2 - 1).astype(np.float32)
w3 = (rng.random([32, 32, 3, 3]) * 2 - 1).astype(np.float32)
w2 = (rng.random([128, 32, 1, 1]) * 2 - 1).astype(np.float32)
res_nodes = [
    helper.make_node("Conv", ["x", "w1"], ["c1"], kernel_shape=[1, 1], name="pw1"),
    helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
    helper.make_node("Conv", ["r1", "w3"], ["c3"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="spconv"),
    helper.make_node("Relu", ["c3"], ["r3"], name="relu3"),
    helper.make_node("Conv", ["r3", "w2"], ["c2"], kernel_shape=[1, 1], name="pw2"),
    helper.make_node("Add", ["c2", "x"], ["summed"], name="residual"),
    helper.make_node("Relu", ["summed"], ["y"], name="reluout"),
]
save_graph("tests/Lokad.Onnx.Bench/oneop/resblock_rn50", res_nodes,
     [tinfo("x", [1, 128, 28, 28])], [tinfo("y", [1, 128, 28, 28])],
     [finit("w1", w1), finit("w3", w3), finit("w2", w2)])

# 30. resnet stem 7x7 stride-2: 1x3x224x224 -> 64x112x112, no bias (C01 family).
w = (rng.random([64, 3, 7, 7]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_7x7_stem",
     helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[7, 7], strides=[2, 2], pads=[3, 3, 3, 3]),
     [tinfo("x", [1, 3, 224, 224])], [tinfo("y", [1, 64, 112, 112])],
     [finit("w", w)])

# 31. resnet stride-2 pointwise, largest shortcut (1024->2048 @14x14), no bias.
w = (rng.random([2048, 1024, 1, 1]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_1x1_s2_2048",
     helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[1, 1], strides=[2, 2]),
     [tinfo("x", [1, 1024, 14, 14])], [tinfo("y", [1, 2048, 7, 7])],
     [finit("w", w)])

# 32. resnet stride-2 spatial, largest downsampler (512ch @14x14), no bias.
w = (rng.random([512, 512, 3, 3]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_3x3_s2_512",
     helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
     [tinfo("x", [1, 512, 14, 14])], [tinfo("y", [1, 512, 7, 7])],
     [finit("w", w)])

# 33. resnet late-stage spatial (512ch @7x7 stride-1), no bias.
w = (rng.random([512, 512, 3, 3]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_3x3_512_7",
     helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
     [tinfo("x", [1, 512, 7, 7])], [tinfo("y", [1, 512, 7, 7])],
     [finit("w", w)])

# 34. resnet 3x3 stride-1 64ch @56x56 (layer1, x3), no bias (C01 family).
w = (rng.random([64, 64, 3, 3]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_3x3_64_56",
     helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
     [tinfo("x", [1, 64, 56, 56])], [tinfo("y", [1, 64, 56, 56])],
     [finit("w", w)])

# 35. resnet 3x3 stride-1 128ch @28x28 (layer2, x3), no bias (C01 family).
w = (rng.random([128, 128, 3, 3]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_3x3_128_28",
     helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
     [tinfo("x", [1, 128, 28, 28])], [tinfo("y", [1, 128, 28, 28])],
     [finit("w", w)])

# 36. resnet 3x3 stride-1 256ch @14x14 (layer3, x5), no bias (C01 family).
w = (rng.random([256, 256, 3, 3]) * 2 - 1).astype(np.float32)
save("tests/Lokad.Onnx.Bench/oneop/conv_3x3_256_14",
     helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
     [tinfo("x", [1, 256, 14, 14])], [tinfo("y", [1, 256, 14, 14])],
     [finit("w", w)])

# 37. e5 MLP up-projection with bias plus exact GELU (F01a region fixture):
# exercises the BiasGelu fused path end to end with agreement gating.
wmlp = (rng.random([384, 1536]) * 2 - 1).astype(np.float32)
bmlp = (rng.random([1536]) * 2 - 1).astype(np.float32)
mlp_nodes = [
    helper.make_node("MatMul", ["x", "wm"], ["mm"], name="uproj"),
    helper.make_node("Add", ["mm", "bm"], ["biased"], name="bias"),
    helper.make_node("Gelu", ["biased"], ["y"], name="act"),
]
save_graph("tests/Lokad.Onnx.Bench/oneop/mlpbias_e5_30", mlp_nodes,
     [tinfo("x", [1, 30, 384])], [tinfo("y", [1, 30, 1536])],
     [finit("wm", wmlp), finit("bm", bmlp)], opset=20)

# 38. e5 MLP region at 8 tokens (e5-8tok core): same fused path, small-M behavior.
mlp8_nodes = [
    helper.make_node("MatMul", ["x", "wm8"], ["mm"], name="uproj"),
    helper.make_node("Add", ["mm", "bm8"], ["biased"], name="bias"),
    helper.make_node("Gelu", ["biased"], ["y"], name="act"),
]
save_graph("tests/Lokad.Onnx.Bench/oneop/mlpbias_e5_8", mlp8_nodes,
     [tinfo("x", [1, 8, 384])], [tinfo("y", [1, 8, 1536])],
     [finit("wm8", wmlp), finit("bm8", bmlp)], opset=20)


# 35. chain fixtures for G03 marginal-node-cost slopes: 1/2/4/8 identical nodes
# per family so session time against node count splits kernel slope from fixed
# intercept. MatMul chain weights are scaled to keep chained values in range.
for _n in [1, 2, 4, 8]:
    _nodes, _prev = [], "x"
    for _i in range(_n):
        _out = "y" if _i == _n - 1 else "s%d" % _i
        _nodes.append(helper.make_node("Softmax", [_prev], [_out], axis=-1))
        _prev = _out
    save_graph("tests/Lokad.Onnx.Bench/oneop/chain_softmax_%d" % _n, _nodes,
         [tinfo("x", [12, 30, 30])], [tinfo("y", [12, 30, 30])], [])

for _n in [1, 2, 4, 8]:
    _nodes, _prev, _inits = [], "a", []
    _dims = [1, 8, 1536]
    for _i in range(_n):
        _bname = "b%d" % _i
        _kdims = [1536, 384] if _dims[-1] == 1536 else [384, 1536]
        _bw = (rng.random(_kdims) * 2 - 1).astype(np.float32) * 0.05
        _inits.append(finit(_bname, _bw))
        _out = "y" if _i == _n - 1 else "m%d" % _i
        _nodes.append(helper.make_node("MatMul", [_prev, _bname], [_out]))
        _prev = _out
        _dims = [1, 8, _kdims[-1]]
    save_graph("tests/Lokad.Onnx.Bench/oneop/chain_matmul_%d" % _n, _nodes,
         [tinfo("a", [1, 8, 1536])], [tinfo("y", _dims)], _inits)

for _n in [1, 2, 4, 8]:
    _nodes, _prev = [], "x"
    _dims = [1, 30, 12, 32]
    for _i in range(_n):
        _out = "y" if _i == _n - 1 else "t%d" % _i
        _nodes.append(helper.make_node("Transpose", [_prev], [_out], perm=[0, 2, 1, 3]))
        _prev = _out
        _dims = [_dims[0], _dims[2], _dims[1], _dims[3]]
    save_graph("tests/Lokad.Onnx.Bench/oneop/chain_transpose_%d" % _n, _nodes,
         [tinfo("x", [1, 30, 12, 32])], [tinfo("y", _dims)], [])

# 39. e5-30 attention scores with RUNTIME scale and causal mask (E04 region):
# mirrors the GPT-2 scale-plus-transpose-plus-mask region at e5-30 shape with
# both MatMul operands live (no constant B), so packing-unavailable traffic is
# priced instead of the constant-B tiles above. Mask is causal 0/-1e4.
_mask = np.zeros([1, 1, 30, 30], dtype=np.float32)
for _i in range(30):
    _mask[0, 0, _i, _i + 1:] = -10000.0
mask_nodes = [
    helper.make_node("MatMul", ["q", "kt"], ["scores"], name="scores"),
    helper.make_node("Mul", ["scores", "scale"], ["scaled"], name="rscale"),
    helper.make_node("Add", ["scaled", "mask"], ["masked"], name="maskadd"),
    helper.make_node("Softmax", ["masked"], ["probs"], axis=-1, name="softmax"),
    helper.make_node("MatMul", ["probs", "v"], ["y"], name="context"),
]
save_graph("tests/Lokad.Onnx.Bench/oneop/attnmask_e5_30", mask_nodes,
     [tinfo("q", [1, 12, 30, 32]), tinfo("kt", [1, 12, 32, 30]),
      tinfo("v", [1, 12, 30, 32]), tinfo("scale", []),
      tinfo("mask", [1, 1, 30, 30])],
     [tinfo("y", [1, 12, 30, 32])], [])

# 40. DINOv3-201 attention block (E04 region): same pattern as #28 with
# S=201/H=6/d=384 and Div-by-constant scale, matching the DINOv3 graph which
# has no mask. Prices the ~46MB score traffic A02 omitted.
# Block weights are scaled to +-0.125 like gemm_512x4608 above: with raw +-1
# draws the region amplifies fp-ordering noise about 100x (6e-06 projection
# diffs grow through scores/softmax/context/outproj to 1.35e-02 at y, measured
# per-prefix against ORT), drowning the 1e-4 gate without changing shapes,
# draws, or timing. Scaled draws agree at 3.2e-05 with margin to spare.
wq6 = ((rng.random([384, 384]) * 2 - 1) * 0.125).astype(np.float32)
wk6 = ((rng.random([384, 384]) * 2 - 1) * 0.125).astype(np.float32)
wv6 = ((rng.random([384, 384]) * 2 - 1) * 0.125).astype(np.float32)
wo6 = ((rng.random([384, 384]) * 2 - 1) * 0.125).astype(np.float32)
shape201 = np.array([1, 201, 6, 64], dtype=np.int64)
merge201 = np.array([1, 201, 384], dtype=np.int64)
dino_nodes = [
    helper.make_node("MatMul", ["x", "wq"], ["q3"], name="qproj"),
    helper.make_node("MatMul", ["x", "wk"], ["k3"], name="kproj"),
    helper.make_node("MatMul", ["x", "wv"], ["v3"], name="vproj"),
    helper.make_node("Reshape", ["q3", "qshape"], ["q4"], name="qreshape"),
    helper.make_node("Reshape", ["k3", "kshape"], ["k4"], name="kreshape"),
    helper.make_node("Reshape", ["v3", "vshape"], ["v4"], name="vreshape"),
    helper.make_node("Transpose", ["q4"], ["q"], perm=[0, 2, 1, 3], name="qtranspose"),
    helper.make_node("Transpose", ["k4"], ["kt"], perm=[0, 2, 3, 1], name="ktranspose"),
    helper.make_node("Transpose", ["v4"], ["v"], perm=[0, 2, 1, 3], name="vtranspose"),
    helper.make_node("MatMul", ["q", "kt"], ["scores"], name="scores"),
    helper.make_node("Div", ["scores", "scale"], ["scaled"], name="scalediv"),
    helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1, name="softmax"),
    helper.make_node("MatMul", ["probs", "v"], ["ctx4"], name="context"),
    helper.make_node("Transpose", ["ctx4"], ["ctx3"], perm=[0, 2, 1, 3], name="ctxmerge"),
    helper.make_node("Reshape", ["ctx3", "yshape"], ["merged"], name="mergereshape"),
    helper.make_node("MatMul", ["merged", "wo"], ["y"], name="outproj"),
]
save_graph("tests/Lokad.Onnx.Bench/oneop/attnblock_dino_201", dino_nodes,
     [tinfo("x", [1, 201, 384])], [tinfo("y", [1, 201, 384])],
     [finit("wq", wq6), finit("wk", wk6), finit("wv", wv6), finit("wo", wo6),
      finit("scale", scale),
      helper.make_tensor("qshape", TensorProto.INT64, [4], shape201),
      helper.make_tensor("kshape", TensorProto.INT64, [4], shape201),
      helper.make_tensor("vshape", TensorProto.INT64, [4], shape201),
      helper.make_tensor("yshape", TensorProto.INT64, [3], merge201)])

# 41. ResNet layer2.0 transition bottleneck at true widths (E04 region):
# 1x1 s1 256->128, 3x3 s2 128->128, 1x1 s1 128->512, plus the 1x1 s2 256->512
# downsample shortcut with residual add and Relu epilogues. Replaces the
# stride-1 proxy with the actual stride-2 family.
# Transition weights are scaled to +-0.25 like gemm_512x4608 above: raw +-1
# draws amplify fp-ordering noise through the pw/spatial/pw chain to 5.1e-04
# at y (measured per-prefix against ORT); scaled draws agree at 1.3e-05 with
# margin, same shapes, draws, and timing.
t1 = ((rng.random([128, 256, 1, 1]) * 2 - 1) * 0.25).astype(np.float32)
t3 = ((rng.random([128, 128, 3, 3]) * 2 - 1) * 0.25).astype(np.float32)
t2 = ((rng.random([512, 128, 1, 1]) * 2 - 1) * 0.25).astype(np.float32)
ts = ((rng.random([512, 256, 1, 1]) * 2 - 1) * 0.25).astype(np.float32)
trans_nodes = [
    helper.make_node("Conv", ["x", "w1"], ["c1"], kernel_shape=[1, 1], name="pw1"),
    helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
    helper.make_node("Conv", ["r1", "w3"], ["c3"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], name="spconv"),
    helper.make_node("Relu", ["c3"], ["r3"], name="relu3"),
    helper.make_node("Conv", ["r3", "w2"], ["c2"], kernel_shape=[1, 1], name="pw2"),
    helper.make_node("Conv", ["x", "ws"], ["ds"], kernel_shape=[1, 1], strides=[2, 2], name="shortcut"),
    helper.make_node("Add", ["c2", "ds"], ["summed"], name="residual"),
    helper.make_node("Relu", ["summed"], ["y"], name="reluout"),
]
save_graph("tests/Lokad.Onnx.Bench/oneop/resblock_s2_trans", trans_nodes,
     [tinfo("x", [1, 256, 56, 56])], [tinfo("y", [1, 512, 28, 28])],
     [finit("w1", t1), finit("w3", t3), finit("w2", t2), finit("ws", ts)])

# 42. two-live-input attention-scores MatMul (E04): 1x12x30x32 @ 1x12x32x30
# with zero constants, pricing the activation-by-activation regime that the
# constant-B attn tiles above cannot show.
save_graph("tests/Lokad.Onnx.Bench/oneop/matmul_runtime_ab",
     [helper.make_node("MatMul", ["a", "b"], ["y"])],
     [tinfo("a", [1, 12, 30, 32]), tinfo("b", [1, 12, 32, 30])],
     [tinfo("y", [1, 12, 30, 30])], [])