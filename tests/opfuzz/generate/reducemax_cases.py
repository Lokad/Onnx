"""ReduceMax differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then append the new files to
tests/opfuzz/corpus/SHA256SUMS (the manifest is append-ordered, not sorted).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Only ReduceSum/ReduceMean had frozen differentials; these
guard axis reduction, negative-axis keepdims-0, and the full-reduction
scalar path against ORT 1.29 in all three runner modes.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    # Row-wise maxima keep a trailing singleton.
    x = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float32)
    node = helper.make_node("ReduceMax", ["x"], ["z"], axes=[1], keepdims=1)
    emit("reducemax_basic", node, [("x", [2, 3])], [("z", [2, 1])], {"x": x})
    # Negative axis without keepdims, mixed signs select per-slice maxima.
    x = (np.arange(12, dtype=np.float32).reshape(2, 2, 3) % 7) - 3.0
    node = helper.make_node("ReduceMax", ["x"], ["z"], axes=[-1], keepdims=0)
    emit("reducemax_negaxis", node, [("x", [2, 2, 3])], [("z", [2, 2])], {"x": x})
    # Full reduction with no axes attribute yields a scalar.
    x = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float32)
    node = helper.make_node("ReduceMax", ["x"], ["z"], keepdims=0)
    emit("reducemax_full", node, [("x", [2, 3])], [("z", [])], {"x": x})
    # int8/uint8 maxima (kernels added for ORT parity; int16/uint16 stay
    # refused on both sides).
    dt8 = {"x": TensorProto.INT8, "z": TensorProto.INT8}
    fd8 = {"x": np.int8}
    node = helper.make_node("ReduceMax", ["x"], ["z"], axes=[0], keepdims=1)
    emit("reducemax_int8", node, [("x", [4])], [("z", [1])],
         {"x": np.array([-128, 127, 1, 2], dtype=np.int8)},
         dtypes=dt8, feed_dtypes=fd8)
    du8 = {"x": TensorProto.UINT8, "z": TensorProto.UINT8}
    fu8 = {"x": np.uint8}
    node = helper.make_node("ReduceMax", ["x"], ["z"], axes=[0], keepdims=1)
    emit("reducemax_uint8", node, [("x", [4])], [("z", [1])],
         {"x": np.array([0, 255, 1, 2], dtype=np.uint8)},
         dtypes=du8, feed_dtypes=fu8)


if __name__ == "__main__":
    main()