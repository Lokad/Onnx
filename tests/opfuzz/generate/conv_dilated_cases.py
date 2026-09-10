"""Dilated Conv differential case (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the conv neighbors.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Dilation spreads the 3x3 kernel over a 5x5 footprint on a
7x7 frame (3x3 output); fuzz-verified class, first frozen differential.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    x = np.arange(49, dtype=np.float32).reshape(1, 1, 7, 7)
    w = (np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) % 3) - 1.0
    w_init = helper.make_tensor("w", TensorProto.FLOAT, [2, 1, 3, 3], w.astype(np.float32))
    node = helper.make_node("Conv", ["x", "w"], ["z"], kernel_shape=[3, 3], dilations=[2, 2])
    emit("conv_dilated", node, [("x", [1, 1, 7, 7])], [("z", [1, 2, 3, 3])],
         {"x": x}, inits=[w_init])


if __name__ == "__main__":
    main()
