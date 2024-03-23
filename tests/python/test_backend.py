import os
import onnx
import pytest

from interop import backend

file_dir = os.path.dirname(os.path.realpath(__file__))

node = onnx.helper.make_node(
            "Squeeze",
            inputs=["x", "axes"],
            outputs=["y"],
)

def test_backend():
    onnx_model_file = os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx")
    rep = backend.prepare_file(onnx_model_file)
    assert rep.graph.Nodes.Count == 12
    o = rep.run([])

