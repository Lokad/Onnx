"""Double-to-int64 Cast differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the add/concat neighbors (first cast cases in the corpus).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. int64 output: the text tensor format and the lane compare
int64 exactly. NaN and infinities are inputs only (the lane text format
carries them since the non-finite-input extension); outputs are finite
int64, which the generator requires.
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
    # Overflow, NaN and infinities saturate; exact powers of two at the
    # int64 boundary behave per the pinned policy.
    node = helper.make_node("Cast", ["x"], ["z"], to=TensorProto.INT64)
    emit("cast_double_sat_int64", node, [("x", [8])], [("z", [8])],
         {"x": np.array([float("nan"), float("inf"), float("-inf"),
                         9.3e18, -9.3e18, 9223372036854775808.0,
                         -9223372036854775808.0, 9223372036854774784.0],
                        dtype=np.float64)},
         dtypes=DT, feed_dtypes=FD)
    # float32 to int32 saturation mirrors the double/int64 sibling.
    node = helper.make_node("Cast", ["x"], ["z"], to=TensorProto.INT32)
    emit("cast_float_sat_int32", node, [("x", [7])], [("z", [7])],
         {"x": np.array([float("nan"), float("inf"), float("-inf"),
                         1e20, -1e20, 2147483648.0, -2147483649.0],
                        dtype=np.float32)},
         dtypes={"x": TensorProto.FLOAT, "z": TensorProto.INT32},
         feed_dtypes={"x": np.float32})
    # Fractions truncate toward zero.
    node = helper.make_node("Cast", ["x"], ["z"], to=TensorProto.INT64)
    emit("cast_double_trunc_int64", node, [("x", [5])], [("z", [5])],
         {"x": np.array([1.9, -1.9, 2.5, -2.5, 0.0], dtype=np.float64)},
         dtypes=DT, feed_dtypes=FD)


if __name__ == "__main__":
    main()