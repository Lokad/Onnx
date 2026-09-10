"""Where differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
after the unsqueeze group (first where cases in the corpus).
Fixed values only, no shared RNG draws, so existing cases are unaffected
by construction. The bool condition rides the extended lane text format.
where_hirank exercises the multidirectional-broadcast fix differentially.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper

DT = {"c": TensorProto.BOOL, "x": TensorProto.FLOAT,
      "y": TensorProto.FLOAT, "z": TensorProto.FLOAT}
FD = {"c": np.bool_}


def main():
    node = helper.make_node("Where", ["c", "x", "y"], ["z"])
    emit("where_basic", node, [("c", [2]), ("x", [2]), ("y", [2])], [("z", [2])],
         {"c": np.array([True, False]),
          "x": np.array([1.0, 2.0], dtype=np.float32),
          "y": np.array([10.0, 20.0], dtype=np.float32)},
         dtypes=DT, feed_dtypes=FD)
    node = helper.make_node("Where", ["c", "x", "y"], ["z"])
    emit("where_hirank", node, [("c", [2, 2]), ("x", [2]), ("y", [2])], [("z", [2, 2])],
         {"c": np.array([[True, False], [False, True]]),
          "x": np.array([1.0, 2.0], dtype=np.float32),
          "y": np.array([10.0, 20.0], dtype=np.float32)},
         dtypes=DT, feed_dtypes=FD)


    node = helper.make_node("Where", ["c", "x", "y"], ["z"])
    emit("where_inf_unselected", node, [("c", [3]), ("x", [3]), ("y", [3])], [("z", [3])],
         {"c": np.array([True, True, True]),
          "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
          "y": np.array([float("inf"), float("-inf"), float("nan")], dtype=np.float32)},
         dtypes=DT, feed_dtypes=FD)
    node = helper.make_node("Where", ["c", "x", "y"], ["z"])
    emit("where_inf_unselected2", node, [("c", [3]), ("x", [3]), ("y", [3])], [("z", [3])],
         {"c": np.array([False, False, False]),
          "x": np.array([float("inf"), float("-inf"), float("nan")], dtype=np.float32),
          "y": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
         dtypes=DT, feed_dtypes=FD)

if __name__ == "__main__":
    main()