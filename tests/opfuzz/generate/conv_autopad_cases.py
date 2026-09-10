"""auto_pad Conv differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the conv neighbors.
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. No corpus model uses auto_pad; these lock the asymmetric
pad distribution (SAME_UPPER pads the end, SAME_LOWER the begin) on a 6x6
frame with kernel 3 stride 2 (3x3 output) against ORT 1.29.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    x = np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6)
    w = (np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3) % 3) - 1.0
    w_init = helper.make_tensor("w", TensorProto.FLOAT, [2, 1, 3, 3], w.astype(np.float32))
    node = helper.make_node("Conv", ["x", "w"], ["z"], kernel_shape=[3, 3],
                            strides=[2, 2], auto_pad="SAME_UPPER")
    emit("conv_same_upper", node, [("x", [1, 1, 6, 6])], [("z", [1, 2, 3, 3])],
         {"x": x}, inits=[w_init])
    node = helper.make_node("Conv", ["x", "w"], ["z"], kernel_shape=[3, 3],
                            strides=[2, 2], auto_pad="SAME_LOWER")
    emit("conv_same_lower", node, [("x", [1, 1, 6, 6])], [("z", [1, 2, 3, 3])],
         {"x": x}, inits=[w_init])


if __name__ == "__main__":
    main()
