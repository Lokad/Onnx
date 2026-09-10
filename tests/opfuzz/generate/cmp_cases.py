"""Equal/Less broadcast differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the div/erf and layernorm/matmul neighbors (first equal/less cases
in the corpus).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. Bool outputs ride the extended lane text format and compare
exactly. Mirrors the unit-pinned broadcast comparison values.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

DT = {"z": TensorProto.BOOL}


def main():
    x = np.array([[1.0], [2.0]], dtype=np.float32)
    y = np.array([10.0, 20.0], dtype=np.float32)
    node = helper.make_node("Equal", ["x", "y"], ["z"])
    emit("equal_broadcast", node, [("x", [2, 1]), ("y", [2])], [("z", [2, 2])],
         {"x": x, "y": y}, dtypes=DT)
    node = helper.make_node("Less", ["x", "y"], ["z"])
    emit("less_broadcast", node, [("x", [2, 1]), ("y", [2])], [("z", [2, 2])],
         {"x": x, "y": y}, dtypes=DT)


    # NaN inputs compare false; the lane carries the non-finite inputs since
    # the format extension, the bool output compares exactly.
    node = helper.make_node("Equal", ["x", "y"], ["z"])
    emit("equal_nan", node, [("x", [3]), ("y", [3])], [("z", [3])],
         {"x": np.array([float("nan"), float("nan"), 1.0], dtype=np.float32),
          "y": np.array([float("nan"), 1.0, float("nan")], dtype=np.float32)},
         dtypes=DT)

    # NaN inputs compare false under Less too (unit-pinned); bool output exact.
    node = helper.make_node("Less", ["x", "y"], ["z"])
    emit("less_nan", node, [("x", [3]), ("y", [3])], [("z", [3])],
         {"x": np.array([float("nan"), float("nan"), 1.0], dtype=np.float32),
          "y": np.array([float("nan"), 1.0, float("nan")], dtype=np.float32)},
         dtypes=DT)

    xb = np.array([[1], [2]])
    yb = np.array([10, 20])
    for suffix, dt_in, cast in (("int64", TensorProto.INT64, lambda a: a.astype(np.int64)),
                                ("double", TensorProto.DOUBLE, lambda a: a.astype(np.float64))):
        dtb = {"x": dt_in, "y": dt_in, "z": TensorProto.BOOL}
        fdb = {"x": np.int64 if suffix == "int64" else np.float64,
               "y": np.int64 if suffix == "int64" else np.float64}
        node = helper.make_node("Equal", ["x", "y"], ["z"])
        emit("equal_broadcast_" + suffix, node, [("x", [2, 1]), ("y", [2])], [("z", [2, 2])],
             {"x": cast(xb), "y": cast(yb)}, dtypes=dtb, feed_dtypes=fdb)
        node = helper.make_node("Less", ["x", "y"], ["z"])
        emit("less_broadcast_" + suffix, node, [("x", [2, 1]), ("y", [2])], [("z", [2, 2])],
             {"x": cast(xb), "y": cast(yb)}, dtypes=dtb, feed_dtypes=fdb)
    node = helper.make_node("Equal", ["x", "y"], ["z"])
    emit("equal_int8", node, [("x", [3]), ("y", [3])], [("z", [3])],
         {"x": np.array([1, 2, -3], dtype=np.int8),
          "y": np.array([1, 0, -3], dtype=np.int8)},
         dtypes={"x": TensorProto.INT8, "y": TensorProto.INT8, "z": TensorProto.BOOL},
         feed_dtypes={"x": np.int8, "y": np.int8})
    node = helper.make_node("Less", ["x", "y"], ["z"])
    emit("less_uint16", node, [("x", [3]), ("y", [3])], [("z", [3])],
         {"x": np.array([1, 2, 60000], dtype=np.uint16),
          "y": np.array([1, 3, 2], dtype=np.uint16)},
         dtypes={"x": TensorProto.UINT16, "y": TensorProto.UINT16, "z": TensorProto.BOOL},
         feed_dtypes={"x": np.uint16, "y": np.uint16})
    node = helper.make_node("Equal", ["x", "y"], ["z"])
    emit("equal_uint32", node, [("x", [3]), ("y", [3])], [("z", [3])],
         {"x": np.array([1, 2, 4294967295], dtype=np.uint32),
          "y": np.array([1, 0, 4294967295], dtype=np.uint32)},
         dtypes={"x": TensorProto.UINT32, "y": TensorProto.UINT32, "z": TensorProto.BOOL},
         feed_dtypes={"x": np.uint32, "y": np.uint32})

    node = helper.make_node("Equal", ["x", "y"], ["z"])
    emit("equal_inf", node, [("x", [4]), ("y", [4])], [("z", [4])],
         {"x": np.array([float("inf"), float("-inf"), 1.0, 2.0], dtype=np.float32),
          "y": np.array([float("inf"), float("-inf"), 1.0, float("-inf")], dtype=np.float32)},
         dtypes=DT)
    node = helper.make_node("Less", ["x", "y"], ["z"])
    emit("less_inf", node, [("x", [4]), ("y", [4])], [("z", [4])],
         {"x": np.array([float("inf"), float("-inf"), 1.0, 2.0], dtype=np.float32),
          "y": np.array([float("inf"), float("-inf"), 1.0, float("-inf")], dtype=np.float32)},
         dtypes=DT)

if __name__ == "__main__":
    main()
