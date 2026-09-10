"""Duplicate-axis Squeeze differential case (explicit maintenance action).

Appends to tests/opfuzz/corpus without touching existing cases: run with the
generator env (tests/opfuzz/generate/requirements-generator.txt), review the
git diff (new directories only), then re-pin tests/opfuzz/corpus/SHA256SUMS
beside the squeeze neighbors. Fixed values only, no shared RNG draws, so
existing cases are unaffected by construction. ORT deduplicates repeated
squeeze axes (verified (2,1) output for axes [1,1] over [2,1,1]); Lokad
matches. Mirrors the C11 validation-parity probe of 2026-09-10.
"""
import os
import sys

import numpy as np
from onnx import TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import emit, helper


def i64(name, vals):
    a = np.array(vals, dtype=np.int64)
    return helper.make_tensor(name, TensorProto.INT64, list(a.shape), a)


def main():
    # Repeated axis 1 over [2,1,1]: deduplicated to a single squeeze -> [2,1].
    node = helper.make_node("Squeeze", ["x", "axes"], ["z"])
    emit("squeeze_dup", node, [("x", [2, 1, 1])], [("z", [2, 1])],
         {"x": np.array([[[1.0], [2.0]]], dtype=np.float32).reshape(2, 1, 1)},
         inits=[i64("axes", [1, 1])])


if __name__ == "__main__":
    main()
