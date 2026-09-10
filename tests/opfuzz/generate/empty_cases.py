"""Empty-tensor differential cases (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS.
See tests/opfuzz/generate/corpus.py for the frozen-case policy.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def main():
    # Fixed shapes and zero fills only: no shared RNG draws, so existing
    # cases are unaffected by construction.
    zeros20 = np.zeros((2, 0), dtype=np.float32)
    node = helper.make_node("ReduceMean", ["x"], ["z"], axes=[1], keepdims=0)
    emit("reducemean_empty", node, [("x", [2, 0])], [("z", [2])], {"x": zeros20})
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], np.array([1], dtype=np.int64))
    node = helper.make_node("ReduceSum", ["x", "axes"], ["z"], keepdims=0)
    emit("reducesum_empty", node, [("x", [2, 0])], [("z", [2])], {"x": zeros20}, inits=[axes])
    node = helper.make_node("Softmax", ["x"], ["z"], axis=-1)
    emit("softmax_empty", node, [("x", [2, 0])], [("z", [2, 0])], {"x": zeros20})
    node = helper.make_node("MatMul", ["x", "y"], ["z"])
    emit("matmul_zero_k", node, [("x", [2, 0]), ("y", [0, 3])], [("z", [2, 3])],
         {"x": zeros20, "y": np.zeros((0, 3), dtype=np.float32)})

    wones = helper.make_tensor("w", TensorProto.FLOAT, [2, 2, 3, 3], np.ones((2, 2, 3, 3), dtype=np.float32))
    node = helper.make_node("Conv", ["x", "w"], ["z"], kernel_shape=[3, 3])
    emit("conv_empty_batch", node, [("x", [0, 2, 5, 5])], [("z", [0, 2, 3, 3])],
         {"x": np.zeros((0, 2, 5, 5), dtype=np.float32)}, inits=[wones])
    wzero = helper.make_tensor("w", TensorProto.FLOAT, [2, 0, 3, 3], np.zeros((2, 0, 3, 3), dtype=np.float32))
    node = helper.make_node("Conv", ["x", "w"], ["z"], kernel_shape=[3, 3])
    emit("conv_empty_channel", node, [("x", [1, 0, 5, 5])], [("z", [1, 2, 3, 3])],
         {"x": np.zeros((1, 0, 5, 5), dtype=np.float32)}, inits=[wzero])


if __name__ == "__main__":
    main()
