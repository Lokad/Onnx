"""Gemm differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then append the new files to
tests/opfuzz/corpus/SHA256SUMS (the manifest is append-ordered, not sorted).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Gemm had xUnit alpha/beta pins but no frozen differential;
these guard the scale/bias epilogue, both transposes, every C broadcast rank,
and the missing-C path against ORT 1.29.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def finit(name, shape, vals):
    return helper.make_tensor(name, TensorProto.FLOAT, list(shape),
                              np.asarray(vals, dtype=np.float32).reshape(-1))


def main():
    # Full [M,N] bias, unit alpha/beta.
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    c = np.array([[0.5, -0.5], [1.5, 2.5]], dtype=np.float32)
    node = helper.make_node("Gemm", ["a", "b", "c"], ["z"], alpha=1.0, beta=1.0)
    emit("gemm_basic", node, [("a", [2, 3]), ("b", [3, 2])], [("z", [2, 2])],
         {"a": a, "b": b}, inits=[finit("c", [2, 2], c)])
    # Row-vector [N] bias with non-unit alpha/beta.
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    c = np.array([0.25, -0.5, 1.0, 2.0], dtype=np.float32)
    node = helper.make_node("Gemm", ["a", "b", "c"], ["z"], alpha=2.0, beta=0.5)
    emit("gemm_alpha_beta", node, [("a", [3, 2]), ("b", [2, 4])], [("z", [3, 4])],
         {"a": a, "b": b}, inits=[finit("c", [4], c)])
    # Transposed B with scalar bias.
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    node = helper.make_node("Gemm", ["a", "b", "c"], ["z"], alpha=1.0, beta=1.0, transB=1)
    emit("gemm_transpose_b", node, [("a", [2, 3]), ("b", [2, 3])], [("z", [2, 2])],
         {"a": a, "b": b}, inits=[finit("c", [], np.float32(0.5))])
    # Transposed A with column-vector [M,1] bias.
    a = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    c = np.array([[10.0], [20.0]], dtype=np.float32)
    node = helper.make_node("Gemm", ["a", "b", "c"], ["z"], alpha=1.0, beta=1.0, transA=1)
    emit("gemm_transpose_a", node, [("a", [3, 2]), ("b", [3, 2])], [("z", [2, 2])],
         {"a": a, "b": b}, inits=[finit("c", [2, 1], c)])
    # Transposed A and B together: A [3,2] reads as [2,3], B [4,3]
    # reads as [3,4]; spot z[0,0] is 1*1+3*2+5*3+0.5 = 22.5.
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    node = helper.make_node("Gemm", ["a", "b", "c"], ["z"], alpha=1.0, beta=1.0, transA=1, transB=1)
    emit("gemm_transpose_ab", node, [("a", [3, 2]), ("b", [4, 3])], [("z", [2, 4])],
         {"a": a, "b": b}, inits=[finit("c", [], np.float32(0.5))])
    # Missing C entirely: beta is ignored on both engines.
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    node = helper.make_node("Gemm", ["a", "b"], ["z"], alpha=1.0, beta=1.0)
    emit("gemm_nobias", node, [("a", [2, 2]), ("b", [2, 2])], [("z", [2, 2])],
         {"a": a, "b": b})


if __name__ == "__main__":
    main()