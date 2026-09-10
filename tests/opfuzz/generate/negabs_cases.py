"""Integer-min Neg/Abs differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the neg/abs kin groups.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. int64 (not int32): the text tensor format and the lane
comparer support float32/float64/int64, and the lane compares int64 exactly.
Mirrors the freshly pinned C11 integer-min wrap values.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

MIN64 = -9223372036854775808
VALS = np.array([MIN64, -1, 0, 1], dtype=np.int64)
DT = {"x": TensorProto.INT64, "z": TensorProto.INT64}
FD = {"x": np.int64}


def main():
    node = helper.make_node("Neg", ["x"], ["z"])
    emit("neg_min_int64", node, [("x", [4])], [("z", [4])], {"x": VALS},
         dtypes=DT, feed_dtypes=FD)
    node = helper.make_node("Abs", ["x"], ["z"])
    emit("abs_min_int64", node, [("x", [4])], [("z", [4])], {"x": VALS},
         dtypes=DT, feed_dtypes=FD)


if __name__ == "__main__":
    main()