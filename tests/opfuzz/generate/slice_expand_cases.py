"""Slice and Expand differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the shape/softmax and concat/erf neighbors (first slice/expand cases
in the corpus).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Slice indices and expand shapes ride as initializers;
only the data tensor is fed. Mirrors the unit-pinned C11 Slice/Expand
boundary values.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def i64(name, vals):
    a = np.array(vals, dtype=np.int64)
    return helper.make_tensor(name, TensorProto.INT64, list(a.shape), a)


def main():
    x = np.arange(10, dtype=np.float32)
    node = helper.make_node("Slice", ["x", "s", "e", "a", "t"], ["z"])
    emit("slice_basic", node, [("x", [10])], [("z", [5])], {"x": x},
         inits=[i64("s", [2]), i64("e", [7]), i64("a", [0]), i64("t", [1])])
    node = helper.make_node("Slice", ["x", "s", "e", "a", "t"], ["z"])
    emit("slice_negstep", node, [("x", [10])], [("z", [3])], {"x": x},
         inits=[i64("s", [8]), i64("e", [2]), i64("a", [0]), i64("t", [-2])])
    node = helper.make_node("Slice", ["x", "s", "e", "a", "t"], ["z"])
    emit("slice_hugestep", node, [("x", [5])], [("z", [1])], {"x": np.arange(5, dtype=np.float32)},
         inits=[i64("s", [0]), i64("e", [5]), i64("a", [0]), i64("t", [1099511627776])])
    node = helper.make_node("Expand", ["x", "shape"], ["z"])
    emit("expand_basic", node, [("x", [3, 1])], [("z", [3, 4])],
         {"x": np.array([[1.0], [2.0], [3.0]], dtype=np.float32)},
         inits=[i64("shape", [3, 4])])
    node = helper.make_node("Expand", ["x", "shape"], ["z"])
    emit("expand_zero", node, [("x", [2, 1])], [("z", [2, 0])],
         {"x": np.array([[1.0], [2.0]], dtype=np.float32)},
         inits=[i64("shape", [2, 0])])
    node = helper.make_node("Split", ["x", "s"], ["a", "b"], axis=1)
    emit("split_bool", node, [("x", [1, 4])], [("a", [1, 2]), ("b", [1, 2])],
         {"x": np.array([[True, False, True, False]])},
         inits=[i64("s", [2, 2])],
         dtypes={"x": TensorProto.BOOL, "a": TensorProto.BOOL, "b": TensorProto.BOOL},
         feed_dtypes={"x": np.bool_})
    node = helper.make_node("Split", ["x", "s"], ["a", "b"], axis=1)
    emit("split_uint32", node, [("x", [1, 4])], [("a", [1, 2]), ("b", [1, 2])],
         {"x": np.array([[1, 2, 3, 4294967295]], dtype=np.uint32)},
         inits=[i64("s", [2, 2])],
         dtypes={"x": TensorProto.UINT32, "a": TensorProto.UINT32, "b": TensorProto.UINT32},
         feed_dtypes={"x": np.uint32})


if __name__ == "__main__":
    main()
