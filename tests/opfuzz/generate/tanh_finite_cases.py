"""Finite-input Tanh differential case (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the tanh kin. Fixed values only, no shared RNG draws, so existing
cases are unaffected by construction. Complements tanh_inf (infinite inputs)
with the normal transcendental path in every lane mode. No Constant case is
possible: OpDump requires at least one --input, so input-less models cannot
run in this lane (same class of limitation as view inputs).
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    node = helper.make_node("Tanh", ["x"], ["z"])
    emit("tanh_finite", node, [("x", [4])], [("z", [4])],
         {"x": np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)})


if __name__ == "__main__":
    main()
