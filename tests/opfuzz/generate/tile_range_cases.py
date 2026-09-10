"""Tile and Range differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the pow/reducemean and sub/transpose neighbors (first tile/range
cases in the corpus).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Mirrors the unit-pinned C11 Tile/Range boundary values.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    node = helper.make_node("Tile", ["x", "r"], ["z"])
    rep3 = helper.make_tensor("r", TensorProto.INT64, [1], np.array([3], dtype=np.int64))
    emit("tile_basic", node, [("x", [2])], [("z", [6])],
         {"x": np.array([1.0, 2.0], dtype=np.float32)}, inits=[rep3])
    rep21 = helper.make_tensor("r", TensorProto.INT64, [2], np.array([2, 1], dtype=np.int64))
    node = helper.make_node("Tile", ["x", "r"], ["z"])
    emit("tile_2d", node, [("x", [2, 2])], [("z", [4, 2])],
         {"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}, inits=[rep21])
    dt = {"x": TensorProto.INT64, "y": TensorProto.INT64,
          "d": TensorProto.INT64, "z": TensorProto.INT64}
    fd = {"x": np.int64, "y": np.int64, "d": np.int64}
    node = helper.make_node("Range", ["x", "y", "d"], ["z"])
    emit("range_basic", node, [("x", []), ("y", []), ("d", [])], [("z", [3])],
         {"x": np.int64(0), "y": np.int64(5), "d": np.int64(2)},
         dtypes=dt, feed_dtypes=fd)
    node = helper.make_node("Range", ["x", "y", "d"], ["z"])
    emit("range_down", node, [("x", []), ("y", []), ("d", [])], [("z", [5])],
         {"x": np.int64(5), "y": np.int64(0), "d": np.int64(-1)},
         dtypes=dt, feed_dtypes=fd)
    repb = helper.make_tensor("r", TensorProto.INT64, [2], np.array([2, 1], dtype=np.int64))
    node = helper.make_node("Tile", ["x", "r"], ["z"])
    emit("tile_bool", node, [("x", [1, 2])], [("z", [2, 2])],
         {"x": np.array([[True, False]])},
         inits=[repb],
         dtypes={"x": TensorProto.BOOL, "z": TensorProto.BOOL},
         feed_dtypes={"x": np.bool_})
    dt16 = {"x": TensorProto.INT16, "y": TensorProto.INT16,
            "d": TensorProto.INT16, "z": TensorProto.INT16}
    fd16 = {"x": np.int16, "y": np.int16, "d": np.int16}
    node = helper.make_node("Range", ["x", "y", "d"], ["z"])
    emit("range_int16", node, [("x", []), ("y", []), ("d", [])], [("z", [3])],
         {"x": np.int16(0), "y": np.int16(5), "d": np.int16(2)},
         dtypes=dt16, feed_dtypes=fd16)


if __name__ == "__main__":
    main()
