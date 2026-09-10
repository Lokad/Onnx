"""Finite Pow differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. All references are finite (the generator refuses
non-finite ORT outputs); negative-base fractional powers stay in unit
tests. Mirrors the freshly pinned C11 Pow boundary values.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    # Negative bases with integer exponents plus a fractional exponent on a
    # positive base: [-512, 1024, -0.125, 0.25], all finite.
    node = helper.make_node("Pow", ["x", "y"], ["z"])
    emit("pow_basic", node, [("x", [4]), ("y", [4])], [("z", [4])],
         {"x": np.array([-8.0, 2.0, -2.0, 0.5], dtype=np.float32),
          "y": np.array([3.0, 10.0, -3.0, 2.0], dtype=np.float32)})
    # Row-vector exponent broadcasts over the last dim:
    # [[1, sqrt(2)], [9, 2]].
    node = helper.make_node("Pow", ["x", "y"], ["z"])
    emit("pow_broadcast", node, [("x", [2, 2]), ("y", [2])], [("z", [2, 2])],
         {"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
          "y": np.array([2.0, 0.5], dtype=np.float32)})


if __name__ == "__main__":
    main()