import os
from interop import lokadonnx

file_dir = os.path.dirname(os.path.realpath(__file__))

def test_load_graph():
    g = lokadonnx.load_graph(os.path.join(file_dir, "..", "..", "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx"))
    assert g.Nodes.Count == 12