"""Double-to-int64 Cast differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the add/concat neighbors (first cast cases in the corpus).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. int64 output: the text tensor format and the lane compare
int64 exactly.
Deliberately truncation-only: a saturation sibling with NaN/inf inputs was
generated and then dropped because the lane text format cannot carry
non-finite inputs (OpDump double.Parse('inf') throws). Saturation stays
pinned in unit tests (CastBoundaryTests), which is the right home for
non-finite inputs.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

DT = {"x": TensorProto.DOUBLE, "z": TensorProto.INT64}
FD = {"x": np.float64}


def main():
    # Fractions truncate toward zero.
    node = helper.make_node("Cast", ["x"], ["z"], to=TensorProto.INT64)
    emit("cast_double_trunc_int64", node, [("x", [5])], [("z", [5])],
         {"x": np.array([1.9, -1.9, 2.5, -2.5, 0.0], dtype=np.float64)},
         dtypes=DT, feed_dtypes=FD)


if __name__ == "__main__":
    main()