"""Old-form version differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the matching op neighbors (slice/squeeze/unsqueeze/reduce/resize/
softmax/split blocks).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Every case below was verified bit-identical tri-mode
(scalar/simd/intrinsics) against ORT 1.29 before freezing: the v10/v11/v13
dispatch arms in src/Lokad.Onnx/Node.cs otherwise run with no frozen
differential coverage (all other corpus cases pin opset 14 or 18).
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
    # Slice-10: starts/ends as inputs, no axes/steps slots.
    x12 = np.arange(8, dtype=np.float32).reshape(2, 4)
    node = helper.make_node("Slice", ["x", "s", "e"], ["z"])
    emit("slice_v10", node, [("x", [2, 4])], [("z", [2, 2])], {"x": x12},
         inits=[i64("s", [0, 1]), i64("e", [2, -1])], opset=10)
    node = helper.make_node("Slice", ["x", "s", "e"], ["z"])
    emit("slice_v10_neg", node, [("x", [2, 4])], [("z", [2, 3])], {"x": x12},
         inits=[i64("s", [-2, -3]), i64("e", [10, 10])], opset=10)
    # Unsqueeze-11 / Squeeze-11: axes as attribute.
    node = helper.make_node("Unsqueeze", ["x"], ["z"], axes=[0, -1])
    emit("unsqueeze_v11", node, [("x", [2, 3])], [("z", [1, 2, 3, 1])],
         {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}, opset=11)
    node = helper.make_node("Squeeze", ["x"], ["z"], axes=[1])
    emit("squeeze_v11", node, [("x", [2, 1, 3])], [("z", [2, 3])],
         {"x": (np.arange(6, dtype=np.float32).reshape(2, 1, 3) - 2.0)}, opset=11)
    # Reduce-attr forms: ReduceSum-11, ReduceMean/ReduceMax-13.
    node = helper.make_node("ReduceSum", ["x"], ["z"], axes=[1], keepdims=0)
    emit("reducesum_v11", node, [("x", [2, 3])], [("z", [2])],
         {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}, opset=11)
    x24 = (np.arange(8, dtype=np.float32).reshape(2, 4) - 3.0)
    node = helper.make_node("ReduceMean", ["x"], ["z"], axes=[0], keepdims=1)
    emit("reducemean_v13", node, [("x", [2, 4])], [("z", [1, 4])], {"x": x24}, opset=13)
    node = helper.make_node("ReduceMax", ["x"], ["z"], axes=[1], keepdims=0)
    emit("reducemax_v13", node, [("x", [2, 4])], [("z", [2])], {"x": x24}, opset=13)
    # Resize-10: scales as the second input (no roi/sizes slots).
    sc = helper.make_tensor("sc", TensorProto.FLOAT, [4], np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
    node = helper.make_node("Resize", ["x", "sc"], ["z"], mode="nearest")
    emit("resize_v10", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)}, inits=[sc], opset=10)
    du8 = {"x": TensorProto.UINT8, "z": TensorProto.UINT8}
    fu8 = {"x": np.uint8}
    node = helper.make_node("Resize", ["x", "sc"], ["z"], mode="nearest")
    emit("resize_v10_uint8", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[0, 1], [2, 3]]]], dtype=np.uint8)}, inits=[sc],
         dtypes=du8, feed_dtypes=fu8, opset=10)
    # Softmax-11: default axis 1 over rank 3 (coerced); Split-11: split attribute.
    node = helper.make_node("Softmax", ["x"], ["z"], axis=1)
    emit("softmax_v11", node, [("x", [2, 3, 4])], [("z", [2, 3, 4])],
         {"x": (np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 11.0)}, opset=11)
    node = helper.make_node("Split", ["x"], ["a", "b"], axis=0, split=[1, 3])
    emit("split_v11", node, [("x", [4, 3])], [("a", [1, 3]), ("b", [3, 3])],
         {"x": (np.arange(12, dtype=np.float32).reshape(4, 3) - 5.0)}, opset=11)


if __name__ == "__main__":
    main()