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

