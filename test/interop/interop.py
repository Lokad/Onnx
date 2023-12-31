import os

from pythonnet import load
import clr

load("coreclr")

clr.AddReference(os.path.join(os.getcwd(), "..", "..", "src", "Lokad.Onnx.Interop", "bin", "Debug", "netstandard2.0", "Lokad.Onnx.Interop.dll"))

from Lokad.Onnx.Interop import Device

b = Device("CPU")

print (b.type)


