import os
import clr
file_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Interop.dll"))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Tensors.dll"))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Backend.dll"))

from Lokad.Onnx import ITensor, ComputationalGraph
from Lokad.Onnx.Interop import Tensors,Graph

def load_graph(file_path:str) -> ComputationalGraph:
    return Graph.LoadFromFile(file_path)