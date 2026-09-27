"""Preserve native bounds and independently bound candidate/current float drift."""
from pathlib import Path
import numpy as np


def compare(candidate, current):
    left, right = Path(candidate).read_bytes(), Path(current).read_bytes()
    assert len(left) == len(right) and len(left) % 4 == 0
    actual, wanted = np.frombuffer(left, dtype='<f4'), np.frombuffer(right, dtype='<f4')
    assert np.isfinite(actual).all() and np.isfinite(wanted).all()
    a, b = actual.astype(np.float64), wanted.astype(np.float64)
    delta = np.abs(a-b)/np.maximum(1., np.abs(b))
    maximum = float(delta.max(initial=0))
    assert maximum <= 1e-4, maximum
    return dict(bit_identical=left == right, maximum_scaled_error=maximum, values=int(actual.size))
