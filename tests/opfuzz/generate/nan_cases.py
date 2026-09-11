"""NaN-output differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside neighboring cases. Fixed values only, no shared RNG draws, so existing
cases are unaffected by construction. These use emit(allow_nonfinite=True):
the finite guard exists to catch RNG blowups, not deliberate NaNs. The lane
compares with equal_nan semantics, so payload differences cannot fail.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

NAN = float("nan")


def main():
    node = helper.make_node("Softmax", ["x"], ["z"], axis=-1)
    emit("nan_softmax", node, [("x", [2, 2])], [("z", [2, 2])],
         {"x": np.array([[NAN, 1.0], [2.0, 3.0]], dtype=np.float32)},
         allow_nonfinite=True)
    node = helper.make_node("Div", ["x", "y"], ["z"])
    emit("nan_div", node, [("x", [2]), ("y", [2])], [("z", [2])],
         {"x": np.array([0.0, 1.0], dtype=np.float32),
          "y": np.array([0.0, 1.0], dtype=np.float32)},
         allow_nonfinite=True)
    node = helper.make_node("Sqrt", ["x"], ["z"])
    emit("nan_sqrt", node, [("x", [2])], [("z", [2])],
         {"x": np.array([-1.0, 4.0], dtype=np.float32)},
         allow_nonfinite=True)


if __name__ == "__main__":
    main()
