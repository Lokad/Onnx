"""GlobalAveragePool differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then append the new files to
tests/opfuzz/corpus/SHA256SUMS (the manifest is append-ordered, not sorted).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. No pooling op had a frozen differential; these guard the
spatial-mean readout over single- and multi-batch frames against ORT 1.29
in all three runner modes. Float only: double GlobalAveragePool has no ORT
CPU kernel, so no double twin can qualify.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    # Two channels pool independently to singletons.
    x = (np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) % 5) - 2.0
    node = helper.make_node("GlobalAveragePool", ["x"], ["z"])
    emit("gap_basic", node, [("x", [1, 2, 2, 3])], [("z", [1, 2, 1, 1])], {"x": x})
    # Two batches with mixed signs and a non-square frame.
    x = (np.arange(24, dtype=np.float32).reshape(2, 1, 4, 3) % 9) - 4.0
    node = helper.make_node("GlobalAveragePool", ["x"], ["z"])
    emit("gap_batched", node, [("x", [2, 1, 4, 3])], [("z", [2, 1, 1, 1])], {"x": x})


if __name__ == "__main__":
    main()