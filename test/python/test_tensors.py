import sys,os

import numpy as np
import pytest

from interop import tensors

def test_tests():
    t = tensors.make_tensor(np.int32)
    #t = DenseTensor[int](3,6,9)

test_tests()
