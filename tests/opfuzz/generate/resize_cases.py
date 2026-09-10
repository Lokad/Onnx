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
    # Integer nearest/linear resizes (kernels added for ORT parity; cubic
    # stays refused there). Linear blends are boundary-safe by
    # construction (quarter-exact weights on small ints); every case below
    # verified bit-identical tri-mode before freezing.
    dt8 = {"x": TensorProto.INT8, "z": TensorProto.INT8}
    fd8 = {"x": np.int8}
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="nearest",
                            coordinate_transformation_mode="half_pixel",
                            nearest_mode="round_prefer_floor")
    emit("resize_nearest_int8", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[-4, -3], [-2, -1]]]], dtype=np.int8)}, inits=[roi, sc2],
         dtypes=dt8, feed_dtypes=fd8)
    du8 = {"x": TensorProto.UINT8, "z": TensorProto.UINT8}
    fu8 = {"x": np.uint8}
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="nearest",
                            coordinate_transformation_mode="half_pixel",
                            nearest_mode="round_prefer_floor")
    emit("resize_nearest_uint8", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[0, 1], [2, 3]]]], dtype=np.uint8)}, inits=[roi, sc2],
         dtypes=du8, feed_dtypes=fu8)
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="linear",
                            coordinate_transformation_mode="half_pixel")
    emit("resize_linear_int8", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[-4, -3], [-2, -1]]]], dtype=np.int8)}, inits=[roi, sc2],
         dtypes=dt8, feed_dtypes=fd8)
    d32 = {"x": TensorProto.INT32, "z": TensorProto.INT32}
    f32 = {"x": np.int32}
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="nearest",
                            coordinate_transformation_mode="half_pixel",
                            nearest_mode="round_prefer_floor")
    emit("resize_nearest_int32", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[0, 1], [2, 3]]]], dtype=np.int32)}, inits=[roi, sc2],
         dtypes=d32, feed_dtypes=f32)
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="linear",
                            coordinate_transformation_mode="asymmetric")
    emit("resize_linear_int32", node, [("x", [1, 1, 2, 2])], [("z", [1, 1, 4, 4])],
         {"x": np.array([[[[0, 1], [2, 3]]]], dtype=np.int32)}, inits=[roi, sc2],
         dtypes=d32, feed_dtypes=f32)

    # Fractional scales: sizes derive from dim*scale in the scale native
    # precision before flooring (6 by float32(7/6) is exactly 7.0f, so the
    # output has 7 rows; verified bit-identical tri-mode before freezing).
    scf = helper.make_tensor("s", TensorProto.FLOAT, [4], np.array([1.0, 1.0, 7.0 / 6.0, 1.0], dtype=np.float32))
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="nearest",
                            coordinate_transformation_mode="half_pixel",
                            nearest_mode="round_prefer_floor")
    emit("resize_frac_nearest", node, [("x", [1, 1, 6, 1])], [("z", [1, 1, 7, 1])],
         {"x": np.array([[[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]]], dtype=np.float32)}, inits=[roi, scf])
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="linear",
                            coordinate_transformation_mode="half_pixel")
    emit("resize_frac_linear", node, [("x", [1, 1, 6, 1])], [("z", [1, 1, 7, 1])],
         {"x": np.array([[[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]]], dtype=np.float32)}, inits=[roi, scf])
    node = helper.make_node("Resize", ["x", "r", "s"], ["z"], mode="nearest",
                            coordinate_transformation_mode="half_pixel",
                            nearest_mode="round_prefer_floor")
    emit("resize_frac_nearest_uint8", node, [("x", [1, 1, 6, 1])], [("z", [1, 1, 7, 1])],
         {"x": np.array([[[[0], [1], [2], [3], [4], [5]]]], dtype=np.uint8)}, inits=[roi, scf],
         dtypes=du8, feed_dtypes=fu8)


if __name__ == "__main__":
    main()
