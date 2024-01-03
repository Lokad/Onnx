import sys,os
import onnx

from pythonnet import load
import clr

import numpy as np
load("coreclr")

file_dir = os.path.dirname(os.path.realpath(__file__))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "netstandard2.0", "publish", "Lokad.Onnx.Tensors.dll"))

from Lokad.Onnx import DenseTensor

def make_tensor(dtype:np.dtype):
    return DenseTensor[int]()
