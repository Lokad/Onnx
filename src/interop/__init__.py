import os
from pythonnet import load, get_runtime_info

file_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
load("coreclr", runtime_config=os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Interop.runtimeconfig.json"))

print(get_runtime_info())