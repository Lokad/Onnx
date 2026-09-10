"""Nearest/linear Resize differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the resize_cubic neighbor (first resize cases beyond cubic in the
corpus).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Mirrors the unit-pinned C11 resampling values: nearest
round_prefer_floor tie selection and finite linear blends.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    roi = helper.make_tensor("r", TensorProto.FLOAT, [8], np.zeros(8, dtype=np.float32))
    sc4 = helper.make_tensor("s", TensorProto.FLOAT, [4], np.array([1, 1, 1, 4], dtype=np.float32))
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="nearest",
                            coordinate_transformation_mode="asymmetric",
                            nearest_mode="round_prefer_floor")
    emit("resize_nearest_tie", node, [("x", [1, 1, 1, 2])], [("z", [1, 1, 1, 8])],
         {"x": np.array([[[[10.0, 20.0]]]], dtype=np.float32)}, inits=[roi, sc4])
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="nearest",
                            coordinate_transformation_mode="half_pixel",
                            nearest_mode="round_prefer_floor")
    emit("resize_nearest_hp", node, [("x", [1, 1, 1, 2])], [("z", [1, 1, 1, 8])],
         {"x": np.array([[[[10.0, 20.0]]]], dtype=np.float32)}, inits=[roi, sc4])
    sc2 = helper.make_tensor("s", TensorProto.FLOAT, [4], np.array([1, 1, 2, 2], dtype=np.float32))
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="linear",
                            coordinate_transformation_mode="asymmetric")
    emit("resize_linear", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)}, inits=[roi, sc2])
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="linear",
                            coordinate_transformation_mode="half_pixel")
    emit("resize_linear_hp", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)}, inits=[roi, sc2])


if __name__ == "__main__":
    main()
