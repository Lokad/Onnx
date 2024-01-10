import os
import clr
import numpy as np


file_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Interop.dll"))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Tensors.dll"))

from System import Array
from Lokad.Onnx import ITensor
from Lokad.Onnx.Interop import Tensors

def make_tensor(dt:np.dtype, *dims) -> ITensor:
    dimsa = Array[int](dims)
    if dt == np.int32:
        return Tensors.MakeTensor[bool](dimsa)
    #elif 

def get_dims(t:ITensor) -> Array[int]: return t.Dims

def broadcast_dim( t:ITensor, dim:int, size:int) -> ITensor: return t.BroadcastDim(dim, size)

a = np.ones([9, 5, 7, 4])
print a[0, 0]