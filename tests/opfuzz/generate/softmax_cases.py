"""Mixed-infinity Softmax differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the softmax neighbors. Fixed values only, no shared RNG draws, so
existing cases are unaffected by construction. All-negative-inf rows yield
NaN (pinned at provider level); mixed -inf lanes yield finite outputs with
exact zeros on the dead lanes (verified tri-mode identical on axis -1 and
within 1 ulp on axis 0 against ORT 1.29).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    # Dead -inf lanes collapse to exact zeros; live lanes renormalize.
    x = np.array([[0.0, float("-inf"), 1.0, float("-inf")],
                  [2.0, 2.0, float("-inf"), 1.0]], dtype=np.float32)
    node = helper.make_node("Softmax", ["x"], ["z"], axis=-1)
    emit("softmax_neginf", node, [("x", [2, 4])], [("z", [2, 4])], {"x": x})
    # Axis-0 variant exercises the strided kernel with dead lanes.
    node = helper.make_node("Softmax", ["x"], ["z"], axis=0)
    emit("softmax_neginf_ax0", node, [("x", [2, 4])], [("z", [2, 4])], {"x": x})


if __name__ == "__main__":
    main()
