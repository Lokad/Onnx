"""Int32 arithmetic-overflow differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the add/sub/mul kin groups.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. int32 inputs/outputs ride the extended lane text format
and compare exactly. Mirrors the unit-pinned wraparound values.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

DT = {"x": TensorProto.INT32, "y": TensorProto.INT32, "z": TensorProto.INT32}
FD = {"x": np.int32, "y": np.int32}
I32 = np.int32


def main():
    node = helper.make_node("Add", ["x", "y"], ["z"])
    emit("add_wrap_int32", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([2147483647, 1000000], dtype=np.int32),
          "y": np.array([1, 1000000], dtype=np.int32)},
         dtypes=DT, feed_dtypes=FD)
    node = helper.make_node("Sub", ["x", "y"], ["z"])
    emit("sub_wrap_int32", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([-2147483648, -1000000], dtype=np.int32),
          "y": np.array([1, 1000000], dtype=np.int32)},
         dtypes=DT, feed_dtypes=FD)
    node = helper.make_node("Mul", ["x", "y"], ["z"])
    emit("mul_wrap_int32", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([1000000, -2147483648], dtype=np.int32),
          "y": np.array([1000000, 2], dtype=np.int32)},
         dtypes=DT, feed_dtypes=FD)


    dt64 = {"x": TensorProto.INT64, "y": TensorProto.INT64, "z": TensorProto.INT64}
    fd64 = {"x": np.int64, "y": np.int64}
    node = helper.make_node("Add", ["x", "y"], ["z"])
    emit("add_wrap_int64", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([9223372036854775807, 3037000500], dtype=np.int64),
          "y": np.array([1, 3037000500], dtype=np.int64)},
         dtypes=dt64, feed_dtypes=fd64)
    node = helper.make_node("Sub", ["x", "y"], ["z"])
    emit("sub_wrap_int64", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([-9223372036854775808, -3037000500], dtype=np.int64),
          "y": np.array([1, 3037000500], dtype=np.int64)},
         dtypes=dt64, feed_dtypes=fd64)
    node = helper.make_node("Mul", ["x", "y"], ["z"])
    emit("mul_wrap_int64", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([3037000500, -9223372036854775808], dtype=np.int64),
          "y": np.array([3037000500, 2], dtype=np.int64)},
         dtypes=dt64, feed_dtypes=fd64)
    du32 = {"x": TensorProto.UINT32, "y": TensorProto.UINT32, "z": TensorProto.UINT32}
    fu32 = {"x": np.uint32, "y": np.uint32}
    node = helper.make_node("Add", ["x", "y"], ["z"])
    emit("add_wrap_uint32", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([4294967295, 1000000], dtype=np.uint32),
          "y": np.array([1, 1000000], dtype=np.uint32)},
         dtypes=du32, feed_dtypes=fu32)
    du64 = {"x": TensorProto.UINT64, "y": TensorProto.UINT64, "z": TensorProto.UINT64}
    fu64 = {"x": np.uint64, "y": np.uint64}
    node = helper.make_node("Mul", ["x", "y"], ["z"])
    emit("mul_wrap_uint64", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([4294967296, 3], dtype=np.uint64),
          "y": np.array([4294967296, 7], dtype=np.uint64)},
         dtypes=du64, feed_dtypes=fu64)
    # int32 MatMul wraps like the scalar lanes on every kernel path
    # (verified tri-mode identical including packed geometries): 2e9 lanes
    # overflow each product, the packed case overflows the accumulation.
    dmm = {"x": TensorProto.INT32, "y": TensorProto.INT32, "z": TensorProto.INT32}
    fmm = {"x": np.int32, "y": np.int32}
    node = helper.make_node("MatMul", ["x", "y"], ["z"])
    emit("imatmul_overflow", node, [("x", [3, 4]), ("y", [4, 3])], [("z", [3, 3])],
         {"x": np.full((3, 4), 2000000000, dtype=np.int32),
          "y": np.full((4, 3), 2, dtype=np.int32)},
         dtypes=dmm, feed_dtypes=fmm)
    node = helper.make_node("MatMul", ["x", "y"], ["z"])
    emit("imatmul_overflow_packed", node, [("x", [32, 17]), ("y", [17, 32])], [("z", [32, 32])],
         {"x": np.full((32, 17), 100000, dtype=np.int32),
          "y": np.full((17, 32), 2, dtype=np.int32)},
         dtypes=dmm, feed_dtypes=fmm)

if __name__ == "__main__":
    main()
