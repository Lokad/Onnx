"""Double-precision differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the matching op neighbors.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Every case below was verified tri-mode against ORT 1.29
before freezing (agreement near 1e-15 against the 1e-5/1e-6 lane gate).
Deliberately absent: double Conv and double Resize, which have no ORT CPU
kernel (NOT_IMPLEMENTED at opsets 11 and 14, probed) and ride unit pins
instead; double Erf/Gelu (same standing, math.erf oracles).
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def fd(name, vals):
    a = np.array(vals, dtype=np.float64)
    return helper.make_tensor(name, TensorProto.DOUBLE, list(a.shape), a)


def main():
    dd = {"x": TensorProto.DOUBLE, "z": TensorProto.DOUBLE}
    ff = {"x": np.float64}
    # Softmax over rows with mixed signs.
    node = helper.make_node("Softmax", ["x"], ["z"], axis=-1)
    emit("softmax_double", node, [("x", [2, 6])], [("z", [2, 6])],
         {"x": (np.arange(12, dtype=np.float64).reshape(2, 6) - 5.0) / 2.0},
         dtypes=dd, feed_dtypes=ff)
    # Rectangular double product.
    da = {"a": TensorProto.DOUBLE, "b": TensorProto.DOUBLE, "z": TensorProto.DOUBLE}
    fa = {"a": np.float64, "b": np.float64}
    node = helper.make_node("MatMul", ["a", "b"], ["z"])
    emit("matmul_double", node, [("a", [4, 5]), ("b", [5, 3])], [("z", [4, 3])],
         {"a": (np.arange(20, dtype=np.float64).reshape(4, 5) - 9.0) / 3.0,
          "b": (np.arange(15, dtype=np.float64).reshape(5, 3) - 7.0) / 3.0},
         dtypes=da, feed_dtypes=fa)
    # Gemm with non-unit alpha/beta and full bias.
    dc = {"a": TensorProto.DOUBLE, "b": TensorProto.DOUBLE,
          "c": TensorProto.DOUBLE, "z": TensorProto.DOUBLE}
    fc = {"a": np.float64, "b": np.float64}
    node = helper.make_node("Gemm", ["a", "b", "c"], ["z"], alpha=1.5, beta=0.5)
    emit("gemm_double", node, [("a", [4, 5]), ("b", [5, 3])], [("z", [4, 3])],
         {"a": (np.arange(20, dtype=np.float64).reshape(4, 5) - 9.0) / 3.0,
          "b": (np.arange(15, dtype=np.float64).reshape(5, 3) - 7.0) / 3.0},
         inits=[fd("c", (np.arange(12, dtype=np.float64).reshape(4, 3) - 5.0) / 2.0)],
         dtypes=dc, feed_dtypes=fc)
    # Bare 2x2 windows over a double frame.
    node = helper.make_node("MaxPool", ["x"], ["z"], kernel_shape=[2, 2], strides=[2, 2])
    emit("maxpool_double", node, [("x", [1, 1, 4, 4])], [("z", [1, 1, 2, 2])],
         {"x": (np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4) - 7.0) / 2.0},
         dtypes=dd, feed_dtypes=ff)
    # Axes-as-input reductions at opset 18 (the exact probed form).
    dx = {"x": TensorProto.DOUBLE, "z": TensorProto.DOUBLE}
    fx = {"x": np.float64}
    node = helper.make_node("ReduceMean", ["x", "ax"], ["z"], keepdims=0)
    emit("reducemean_double", node, [("x", [2, 3, 4])], [("z", [2, 4])],
         {"x": (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 11.0) / 2.0},
         inits=[helper.make_tensor("ax", TensorProto.INT64, [1], np.array([1], dtype=np.int64))],
         dtypes=dx, feed_dtypes=fx, opset=18)
    node = helper.make_node("ReduceMax", ["x", "ax"], ["z"], keepdims=0)
    emit("reducemax_double", node, [("x", [2, 3, 4])], [("z", [2, 4])],
         {"x": (np.arange(24, dtype=np.float64).reshape(2, 3, 4) - 11.0) / 2.0},
         inits=[helper.make_tensor("ax", TensorProto.INT64, [1], np.array([1], dtype=np.int64))],
         dtypes=dx, feed_dtypes=fx, opset=18)


if __name__ == "__main__":
    main()
