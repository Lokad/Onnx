import sys,os

from pythonnet import load
import clr

import numpy as np
load("coreclr")

file_dir = os.path.dirname(os.path.realpath(__file__))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "netstandard2.0", "publish", "Lokad.Onnx.Interop.dll"))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "netstandard2.0", "publish", "Lokad.Onnx.Tensors.dll"))

from System import Array
from Lokad.Onnx import ITensor
from Lokad.Onnx.Interop import Tensors

def make_tensor(dt:np.dtype, *dims) -> ITensor:
    dimsa = Array[int](dims)
    return Tensors.MakeTensor[int](dimsa)

def get_dims(t:ITensor) -> Array[int]: return t.Dims

def broadcast_dim( t:ITensor, dim:int, size:int) -> ITensor: return t.BroadcastDim(dim, size)