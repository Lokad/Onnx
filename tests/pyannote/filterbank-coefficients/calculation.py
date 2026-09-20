import math
from decimal import Decimal, localcontext
from shared import np

def ideal_decimal():
    with localcontext() as ctx:
        ctx.prec = 50
        d = Decimal; low = d(1127) * (1 + d(20) / 700).ln(); high = d(1127) * (1 + d(8000) / 700).ln()
        step = (high - low) / 81
        bins = [d(1127) * (1 + d(16000) * k / (512 * 700)).ln() for k in range(256)]
        rows = []
        for band in range(80):
            left, middle, right = (low + step * j for j in (band, band + 1, band + 2))
            rows.append([float(max(d(0), min((v - left) / (middle - left), (right - v) / (right - middle)))) for v in bins])
        return np.asarray(rows, dtype=np.float64)

def ideal_double():
    lo = 1127 * math.log(1 + 20 / 700); hi = 1127 * math.log(1 + 8000 / 700)
    result = np.zeros((80, 256), dtype=np.float64)
    for k in range(256):
        mel = 1127 * math.log(1 + (16000 * k / 512) / 700)
        position = (mel - lo) / ((hi - lo) / 81)
        for band in range(80): result[band, k] = max(0., 1 - abs(position - (band + 1)))
    return result

def calculate(power, weights):
    assert power.ndim == 2 and power.shape[1] == 257 and weights.shape == (80, 256)
    assert power.dtype == weights.dtype == np.float64 and np.isfinite(power).all() and np.isfinite(weights).all()
    assert (power >= 0).all() and (weights >= 0).all()
    energy = power[:, :256] @ weights.T
    raw = np.log(np.maximum(energy, float(np.finfo(np.float32).eps)))
    return dict(energy=energy, raw=raw, features=(raw - raw.mean(axis=0, keepdims=True))[None])

def scalar_calculate(power, weights):
    energy = np.empty((power.shape[0], 80), dtype=np.float64)
    raw = np.empty_like(energy); features = np.empty_like(energy)
    for band in range(80):
        support = [(int(k), float(weights[band, k])) for k in np.flatnonzero(weights[band])]
        for frame, row in enumerate(power):
            total = math.fsum(float(row[k]) * weight for k, weight in support)
            energy[frame, band] = total
            raw[frame, band] = math.log(max(float(np.finfo(np.float32).eps), total))
        mean = math.fsum(float(v) for v in raw[:, band]) / power.shape[0]
        features[:, band] = raw[:, band] - mean
    return dict(energy=energy, raw=raw, features=features[None])
