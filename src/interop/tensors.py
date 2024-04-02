import os
import ctypes

import numpy as np
from typing import Any, Dict
import clr

file_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Interop.dll"))
clr.AddReference(os.path.join(file_dir, "..", "Lokad.Onnx.Interop", "bin", "Release", "net6.0", "publish", "Lokad.Onnx.Tensors.dll"))

import System
from System import Array, Int32, Single, Double, String
from System.Collections.Generic import Dictionary
from System.Runtime.InteropServices import GCHandle, GCHandleType
from Lokad.Onnx import ITensor
from Lokad.Onnx.Interop import Tensors

_MAP_NP_NET = {
    np.dtype('float32'): System.Single,
    np.dtype('float64'): System.Double,
    np.dtype('int8')   : System.SByte,
    np.dtype('int16')  : System.Int16,
    np.dtype('int32')  : System.Int32,
    np.dtype('int64')  : System.Int64,
    np.dtype('uint8')  : System.Byte,
    np.dtype('uint16') : System.UInt16,
    np.dtype('uint32') : System.UInt32,
    np.dtype('uint64') : System.UInt64,
    np.dtype('bool')   : System.Boolean,
}
_MAP_NET_NP = {
    'Single' : np.dtype('float32'),
    'Double' : np.dtype('float64'),
    'SByte'  : np.dtype('int8'),
    'Int16'  : np.dtype('int16'), 
    'Int32'  : np.dtype('int32'),
    'Int64'  : np.dtype('int64'),
    'Byte'   : np.dtype('uint8'),
    'UInt16' : np.dtype('uint16'),
    'UInt32' : np.dtype('uint32'),
    'UInt64' : np.dtype('uint64'),
    'Boolean': np.dtype('bool'),
}

def asNumpyArray(netArray:Array) -> np.ndarray[Any]:
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

    try: # Memmove 
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__['data'][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated: sourceHandle.Free()
    return npArray

def asNetArray(npArray:np.ndarray[Any]) -> Array:
    '''
    Given a `numpy.ndarray` returns a CLR `System.Array`.  See _MAP_NP_NET for 
    the mapping of Numpy dtypes to CLR types.

    Note: `complex64` and `complex128` arrays are converted to `float32` 
    and `float64` arrays respectively with shape [m,n,...] -> [m,n,...,2]
    '''
    dims = npArray.shape
    dtype = npArray.dtype
    # For complex arrays, we must make a view of the array as its corresponding 
    # float type.
    if dtype == np.complex64:
        dtype = np.dtype('float32')
        dims.append(2)
        npArray = npArray.view(np.float32).reshape(dims)
    elif dtype == np.complex128:
        dtype = np.dtype('float64')
        dims.append(2)
        npArray = npArray.view(np.float64).reshape(dims)

    netDims = Array.CreateInstance(Int32, npArray.ndim)
    for I in range(npArray.ndim):
        netDims[I] = Int32(dims[I])
    
    if not npArray.flags.c_contiguous:
        npArray = npArray.copy(order='C')
    assert npArray.flags.c_contiguous

    try:
        netArray = Array.CreateInstance(_MAP_NP_NET[dtype], netDims)
    except KeyError:
        raise NotImplementedError("asNetArray does not yet support dtype {}".format(dtype))

    try: # Memmove 
        destHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = npArray.__array_interface__['data'][0]
        destPtr = destHandle.AddrOfPinnedObject().ToInt64()
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if destHandle.IsAllocated: destHandle.Free()
    return netArray

def make_tensor_from_ndarray(a:np.ndarray[Any]) -> ITensor:
    if (a.dtype == np.uint8):
        return Tensors.MakeTensor[System.Byte](asNetArray(a))
    elif (a.dtype == np.int8):
        return Tensors.MakeTensor[System.SByte](asNetArray(a))
    elif (a.dtype == np.int16):
        return Tensors.MakeTensor[System.Int16](asNetArray(a))
    elif (a.dtype == np.uint16):
        return Tensors.MakeTensor[System.UInt16](asNetArray(a))
    elif (a.dtype == np.int32):
        return Tensors.MakeTensor[System.Int32](asNetArray(a))
    elif (a.dtype == np.uint32):
        return Tensors.MakeTensor[System.UInt32](asNetArray(a))
    elif (a.dtype == np.int64):
        return Tensors.MakeTensor[System.Int64](asNetArray(a))
    elif (a.dtype == np.uint64):
        return Tensors.MakeTensor[System.UInt64](asNetArray(a))
    elif (a.dtype == np.float32):
        return Tensors.MakeTensor[System.Single](asNetArray(a))
    elif (a.dtype == np.float64):
        return Tensors.MakeTensor[System.Double](asNetArray(a))
    else: raise RuntimeError('fThe type {a.dtype} is not supported.')

def make_ndarray_from_tensor(t:ITensor) -> np.ndarray[Any]:
    a =  asNumpyArray(t.ToArray())
    a.resize(t.Dims)
    return a

def make_empty_tensor(dt:np.dtype, *dims) -> ITensor:
    dimsa = Array[int](*dims)
    if dt == np.int32:
        return Tensors.MakeTensor[int](dimsa)
    #elif 

def make_tensor_array(tensors) -> Array[ITensor]:
    return Array[ITensor](tensors)
    
def make_tensor_dictionary(tensors:Dict[str, ITensor]) -> Dictionary[str, ITensor]:
    dictionary = Dictionary[String, ITensor]()
    for key, value in tensors.items():
        dictionary.Add(key, value)
    return dictionary
    
def ndarray_eq(a:np.ndarray, b:np.ndarray) -> bool:
    return set(a.shape) == set(b.shape) and set(a.flatten()) == set(b.flatten())

def zeros(dt:np.dtype, *dims) -> ITensor:
    dimsa = Array[int](dims)
    if dt == np.int32:
        return Tensors.Zeros[int](dimsa)

def ones(dt:np.dtype, *dims) -> ITensor:
    dimsa = Array[int](dims)
    if dt == np.int32:
        return Tensors.Ones[int](dimsa)

def arange(start:int, stop:int, step:int=1)->ITensor: return Tensors.ARange(start, stop, step)

def get_dims(t:ITensor) -> Array[int]: return t.Dims

def broadcast_dim( t:ITensor, dim:int, size:int) -> ITensor: return t.BroadcastDim(dim, size)

def slice(t: ITensor, *dims) -> ITensor:
    dimsa = Array[int](dims)
    return t.Slice(dimsa)

def add(x:ITensor, y) -> ITensor: return Tensors.Add(x, y)

def matmul(x:ITensor, y) -> ITensor: return Tensors.MatMul(x, y)