"""Sequence-intermediate differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the split kin (first seq blocks in the manifest). Fixed values only, no shared RNG draws, so existing
cases are unaffected by construction. The SplitToSequence/SequenceAt/Relu
chains mirror the DINOv3 sequence usage (unit-pinned roundtrip); sequences
ride as undeclared intermediates between dense tensor inputs and outputs, so
the corpus text format needs no extension. write_model accepts a node list
for these chains; single-node callers are unaffected.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    x = np.array([[-1.5, 2.0, -0.25, 3.0], [0.5, -4.0, 1.25, -2.0]], dtype=np.float32)
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq"], axis=0, keepdims=1),
        helper.make_node("SequenceAt", ["seq", "pos"], ["y"]),
        helper.make_node("Relu", ["y"], ["z"]),
    ]
    emit("seq_at_relu", nodes, [("x", [2, 4]), ("pos", [1])], [("z", [1, 4])],
         {"x": x, "pos": np.array([1], dtype=np.int64)},
         dtypes={"pos": TensorProto.INT64}, feed_dtypes={"pos": np.int64})
    nodes = [
        helper.make_node("SplitToSequence", ["x"], ["seq"], axis=1, keepdims=0),
        helper.make_node("SequenceAt", ["seq", "pos"], ["y"]),
        helper.make_node("Relu", ["y"], ["z"]),
    ]
    emit("seq_at_neg_relu", nodes, [("x", [2, 4]), ("pos", [1])], [("z", [2])],
         {"x": x, "pos": np.array([-1], dtype=np.int64)},
         dtypes={"pos": TensorProto.INT64}, feed_dtypes={"pos": np.int64})


if __name__ == "__main__":
    main()
