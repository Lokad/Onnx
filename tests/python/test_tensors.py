import sys,os

import numpy as np
import pytest

from interop import tensors

def test_make_tensors():
    t = tensors.make_tensor(np.int32, 4, 5, 6)
    d = tensors.get_dims(t)
    assert d.Length == 3
    assert d[0] == 4
    tn = np.ndarray((4,5,6), np.int32)
    assert set(tn.shape) == set(t.Dims)

def test_arange():
    t  = tensors.arange(0, 9).Reshape(3, 3)
    tnd = np.arange(0, 9).reshape(3, 3)
    assert set(t.Dims) == set(tnd.shape)
    assert t[2, 0] == 6
    assert t[0, 1] == tnd[0, 1]
    assert t[2, 0] == tnd[2, 0]
    t = tensors.arange(0, 60, 2).Reshape(3, 5, 2)
    tnd = np.arange(0, 60, 2).reshape(3, 5, 2)
    assert set(t.Dims) == set(tnd.shape)
    assert t[2, 1, 0] == tnd[2, 1, 0]
   
def test_slice():
    t  = tensors.arange(0, 9).Reshape(3, 3)
    tnd = np.arange(0, 9).reshape(3, 3)
    ts = t["1:2","..."]
    tnds = tnd[1:2,...]
    assert set(tnds.shape) == set(ts.Dims)
    assert ts[0, 2] == tnds[0, 2]
    assert set(tnds.flat) == set(ts)


def test_broadcast_dim():
    t = tensors.make_tensor(np.int32, 4, 5, 6, 1)
    d = tensors.get_dims(t)
    assert d.Length == 4
    assert d[3] == 1
    bct = t.BroadcastDim(3, 255)
    d = tensors.get_dims(bct)
    assert set(d) == set(np.broadcast_shapes((4, 5, 6, 1), (4, 5, 6, 255)))
    #assert set(d) == set(np.broadcast_shapes((4, 5, 6, 1), (4, 5, 6, 254)))
    assert d[3] == 255
    t = tensors.make_tensor(np.int32, 4, 5, 1)
    bct = tensors.broadcast_dim(t, 2, 255)
    
def test_add():
     tnd1 = np.arange(0, 27).reshape(3, 3, 3)
     tnd2 = np.arange(0, 27).reshape(3, 3, 3)
     r = np.add(tnd1, tnd2)
     t1 = tensors.arange(0, 27).Reshape(3, 3, 3)
     t2 = tensors.arange(0, 27).Reshape(3, 3, 3)
     rt = tensors.add(t1, t2)
     assert set(r.flat) == set(rt)

