import sys,os

import numpy as np
import pytest

from interop import tensors


def test_make_tensors():
    t = tensors.make_tensor(np.int32, 4, 5, 6)
    print(t["...,"].Rank)

    d = tensors.get_dims(t)
    assert d.Length == 3
    assert d[0] == 4

def test_arange():
    t  = tensors.arange(0, 9).Reshape(3, 3)
    tnd = np.arange(0, 9).reshape(3, 3)
    assert set(t.Dims) == set(tnd.shape)
    assert t[0, 1] == tnd[0, 1]
    assert t[2, 0] == tnd[2, 0]
    t = tensors.arange(0, 60, 2).Reshape(3, 5, 2)
    tnd = np.arange(0, 60, 2).reshape(3, 5, 2)
    assert set(t.Dims) == set(tnd.shape)
    assert t[2, 1, 0] == tnd[2, 1, 0]
    
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
    
      #assert d[3] == 255

