"""Integer reduction-saturation differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the reduction kin groups.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. int32 inputs/outputs ride the extended lane text format
and compare exactly. Mirrors the unit-pinned saturation semantics, including
the discriminating vector that separates exact-clamp from running saturation.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

DT = {"x": TensorProto.INT32, "z": TensorProto.INT32}
FD = {"x": np.int32}


def main():
    node = helper.make_node("ReduceSum", ["x"], ["z"], keepdims=0)
    emit("reduce_sum_sat_int32", node, [("x", [4])], [("z", [])],
         {"x": np.array([2147483647, 2147483647, -2147483648, -2147483648],
                        dtype=np.int32)},
         dtypes=DT, feed_dtypes=FD)
    node = helper.make_node("ReduceMean", ["x"], ["z"], keepdims=0)
    emit("reduce_mean_sat_int32", node, [("x", [2])], [("z", [])],
         {"x": np.array([2147483647, 2147483647], dtype=np.int32)},
         dtypes=DT, feed_dtypes=FD)
    DT64 = {"x": TensorProto.INT64, "z": TensorProto.INT64}
    FD64 = {"x": np.int64}
    node = helper.make_node("ReduceSum", ["x"], ["z"], keepdims=0)
    emit("reducesum_basic_int64", node, [("x", [3])], [("z", [])],
         {"x": np.array([1, 2, 3], dtype=np.int64)},
         dtypes=DT64, feed_dtypes=FD64)
    # int64 accumulates in double then saturates (ORT 1.29): the
    # discriminator below is 0, which wide-then-clamp (-2) and running
    # saturation (Min) both miss; all frozen values are order-robust.
    node = helper.make_node("ReduceSum", ["x"], ["z"], keepdims=0)
    emit("reducesum_sat_int64", node, [("x", [4])], [("z", [])],
         {"x": np.array([9223372036854775807, 9223372036854775807,
                        -9223372036854775808, -9223372036854775808], dtype=np.int64)},
         dtypes=DT64, feed_dtypes=FD64)
    node = helper.make_node("ReduceMean", ["x"], ["z"], keepdims=0)
    emit("reducemean_basic_int64", node, [("x", [4])], [("z", [])],
         {"x": np.array([1, 2, 3, 4], dtype=np.int64)},
         dtypes=DT64, feed_dtypes=FD64)
    # Truncation toward zero discriminates against half-up, half-even
    # (positive) and floor (negative); both frozen values are order-robust.
    node = helper.make_node("ReduceMean", ["x"], ["z"], keepdims=0)
    emit("reducemean_truncpos_int64", node, [("x", [2])], [("z", [])],
         {"x": np.array([1, 2], dtype=np.int64)},
         dtypes=DT64, feed_dtypes=FD64)
    node = helper.make_node("ReduceMean", ["x"], ["z"], keepdims=0)
    emit("reducemean_truncneg_int64", node, [("x", [2])], [("z", [])],
         {"x": np.array([-1, -2], dtype=np.int64)},
         dtypes=DT64, feed_dtypes=FD64)



if __name__ == "__main__":
    main()