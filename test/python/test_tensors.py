import sys,os

import numpy as np
import pytest

from interop import tensors


def test_make_tensors():
    t = tensors.make_tensor(np.int32, 4, 5, 6)
    print(tensors.get_dims(t))

    d = tensors.get_dims(t)
    assert d.Length == 3
    assert d[0] == 4

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
      t = tensors.make_tensor(np.int32, 4, 5)
      #ct = tensors.broadcast_dim()
    
      #assert d[3] == 255

