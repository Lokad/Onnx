"""MaxPool differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then append the new files to
tests/opfuzz/corpus/SHA256SUMS (the manifest is append-ordered, not sorted).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. MaxPool had xUnit value/pad/ceil/inf pins but no frozen
differential; these guard the window readout, padding, multi-channel flow,
and ceil-mode edge shape against ORT 1.29 in all three runner modes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    # Bare 2x2 windows over a 4x4 frame.
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    node = helper.make_node("MaxPool", ["x"], ["z"], kernel_shape=[2, 2], strides=[2, 2])
    emit("maxpool_basic", node, [("x", [1, 1, 4, 4])], [("z", [1, 1, 2, 2])], {"x": x})
    # 3x3 windows, stride 1, explicit one-ring padding preserves the frame.
    x = (np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5) % 7) - 3.0
    node = helper.make_node("MaxPool", ["x"], ["z"], kernel_shape=[3, 3],
                            strides=[1, 1], pads=[1, 1, 1, 1])
    emit("maxpool_pad_stride", node, [("x", [1, 1, 5, 5])], [("z", [1, 1, 5, 5])], {"x": x})
    # Two channels pool independently.
    x = (np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4) % 9) - 4.0
    node = helper.make_node("MaxPool", ["x"], ["z"], kernel_shape=[2, 2], strides=[2, 2])
    emit("maxpool_multichannel", node, [("x", [1, 2, 4, 4])], [("z", [1, 2, 2, 2])], {"x": x})
    # Ceil mode keeps the partial trailing window floor pooling drops (1x1 floor vs 2x2 ceil here).
    x = (np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4) % 5) - 2.0
    node = helper.make_node("MaxPool", ["x"], ["z"], kernel_shape=[3, 3],
                            strides=[2, 2], ceil_mode=1)
    emit("maxpool_ceil", node, [("x", [1, 1, 4, 4])], [("z", [1, 1, 2, 2])], {"x": x})


if __name__ == "__main__":
    main()