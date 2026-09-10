"""Infinite-input, finite-output differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the tanh/erf kin groups.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Only ops whose infinite inputs yield finite outputs qualify
(the generator refuses non-finite references, so Neg/Abs/Relu/Sqrt/Gelu with
infinite outputs stay unit-pinned only); NaN inputs never qualify. Mirrors
the unit-pinned C11 exceptional values and adds scalar/simd/intrinsics mode
coverage the unit pins (Auto mode) do not reach.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

I = float("inf")


def main():
    node = helper.make_node("Tanh", ["x"], ["z"])
    emit("tanh_inf", node, [("x", [3])], [("z", [3])],
         {"x": np.array([I, -I, 0.0], dtype=np.float32)})
    node = helper.make_node("Erf", ["x"], ["z"])
    emit("erf_inf", node, [("x", [3])], [("z", [3])],
         {"x": np.array([I, -I, 0.0], dtype=np.float32)})


if __name__ == "__main__":
    main()
