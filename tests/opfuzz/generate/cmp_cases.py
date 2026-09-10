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

if __name__ == "__main__":
    main()