"""Rank-0 scalar Pow differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the pow kin groups.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Mirrors the unit-pinned rank-0 Pow values; the lane runner
(OpDump) handles rank-0 tensors generically.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    # Fully scalar: (-8)^3 = -512.
    node = helper.make_node("Pow", ["x", "y"], ["z"])
    emit("pow_scalar", node, [("x", []), ("y", [])], [("z", [])],
         {"x": np.array(-8.0, dtype=np.float32),
          "y": np.array(3.0, dtype=np.float32)})
    # Scalar exponent broadcasts: [-8,2,-2,0.5]^3 = [-512,8,-8,0.125].
    node = helper.make_node("Pow", ["x", "y"], ["z"])
    emit("pow_scalar_broadcast", node, [("x", [4]), ("y", [])], [("z", [4])],
         {"x": np.array([-8.0, 2.0, -2.0, 0.5], dtype=np.float32),
          "y": np.array(3.0, dtype=np.float32)})


if __name__ == "__main__":
    main()