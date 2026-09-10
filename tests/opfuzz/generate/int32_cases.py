"""First int32-fed differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the range and neg kin groups.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. int32 inputs/outputs ride the extended lane text format
and compare exactly. Mirrors unit-pinned int32 Range and Neg-min values.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    dt3 = {"x": TensorProto.INT32, "y": TensorProto.INT32,
           "d": TensorProto.INT32, "z": TensorProto.INT32}
    fd3 = {"x": np.int32, "y": np.int32, "d": np.int32}
    node = helper.make_node("Range", ["x", "y", "d"], ["z"])
    emit("range_int32", node, [("x", []), ("y", []), ("d", [])], [("z", [3])],
         {"x": np.int32(0), "y": np.int32(5), "d": np.int32(2)},
         dtypes=dt3, feed_dtypes=fd3)
    dt1 = {"x": TensorProto.INT32, "z": TensorProto.INT32}
    fd1 = {"x": np.int32}
    node = helper.make_node("Neg", ["x"], ["z"])
    emit("neg_min_int32", node, [("x", [4])], [("z", [4])],
         {"x": np.array([-2147483648, -1, 0, 1], dtype=np.int32)},
         dtypes=dt1, feed_dtypes=fd1)


if __name__ == "__main__":
    main()