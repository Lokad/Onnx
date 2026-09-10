"""Cos/Sin differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then append the new files to
tests/opfuzz/corpus/SHA256SUMS (the manifest is append-ordered, not sorted).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Cos/Sin had xUnit basic/exceptional/wide-body pins but no
frozen differential; these guard quadrant angles, signs, and two decades of
argument magnitude against ORT 1.29 in all three runner modes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

ANGLES = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2, 1.0, -1.0, 3.5, -3.5,
                   10.0, -10.0, 100.0, -100.0], dtype=np.float32)


def main():
    node = helper.make_node("Cos", ["x"], ["z"])
    emit("cos_basic", node, [("x", [12])], [("z", [12])], {"x": ANGLES})
    node = helper.make_node("Sin", ["x"], ["z"])
    emit("sin_basic", node, [("x", [12])], [("z", [12])], {"x": ANGLES})


if __name__ == "__main__":
    main()