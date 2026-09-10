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


if __name__ == "__main__":
    main()