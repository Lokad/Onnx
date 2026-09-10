"""Subnormal-input differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the alphabetically neighboring cases.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. All references are finite (subnormal in, subnormal or
normal out), so the cases run in every lane mode. Probed 2026-09-11: ORT
1.29 and Lokad.Onnx agree bit-exactly in scalar, simd, and intrinsics modes;
in particular Mul doubling a subnormal stays subnormal (no flush-to-zero).
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

SUB = np.array([1e-38, -1e-38, 1.4e-45, 0.0], dtype=np.float32)
POS = np.array([1e-38, 1.4e-45, 0.0, 2.4e-38], dtype=np.float32)


def main():
    node = helper.make_node("Add", ["x", "x"], ["z"])
    emit("denorm_add", node, [("x", [4])], [("z", [4])], {"x": SUB})
    node = helper.make_node("Mul", ["x", "y"], ["z"])
    emit("denorm_mul", node, [("x", [4]), ("y", [4])], [("z", [4])],
         {"x": SUB, "y": np.full((4,), 2.0, dtype=np.float32)})
    node = helper.make_node("Sqrt", ["y"], ["z"])
    emit("denorm_sqrt", node, [("y", [4])], [("z", [4])], {"y": POS})
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    node = helper.make_node("ReduceSum", ["x", "axes"], ["z"], keepdims=1)
    emit("denorm_reducesum", node, [("x", [4])], [("z", [1])], {"x": SUB}, inits=[axes])
    node = helper.make_node("ReduceMean", ["x"], ["z"], axes=[0], keepdims=1)
    emit("denorm_reducemean", node, [("x", [4])], [("z", [1])], {"x": POS})
    node = helper.make_node("ReduceMax", ["x"], ["z"], axes=[0], keepdims=1)
    emit("denorm_reducemax", node, [("x", [4])], [("z", [1])], {"x": SUB})
    # Subnormal-magnitude MatMul accumulation: the [0,0] cell differs
    # from ORT by 1 ulp (summation-order class, inside lane tolerance).
    node = helper.make_node("MatMul", ["x", "y"], ["z"])
    emit("denorm_matmul", node, [("x", [2, 2]), ("y", [2, 2])], [("z", [2, 2])],
         {"x": np.array([[1e-38, 2e-38], [3e-38, 0.0]], dtype=np.float32),
          "y": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})


if __name__ == "__main__":
    main()
