"""Fixed local diagnostic: float32 graph, changing only matrix reductions."""
import os
import sys
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT/'tests/whisper/full-reference'
sys.path.insert(0, str(TOOLS))
from common import np, pin, read, write, rel, raw, source, tensor_array, packages, THREADS, absent, MODEL, DATA, ORIGINAL, WEIGHTS
packages()
import psutil
from interpreter import execute, attributes

BASE = ROOT/'artifacts/whisper-matmul-precision-20260921'
FULL = ROOT/'artifacts/whisper-full-reference-20260920'
PROTOCOL = 'whisper-fp32-matmul-accumulation-v1'
SELECTED = [0, 10, 9, 20]
MODES = ['float32', 'wide-matmul']
LIMITS = dict(seconds=900, rss=4*1024**3, available=1024**3, preflight_available=6*1024**3, disk=8*1024**3)


def calculate(node, inputs, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if node.op_type == 'MatMul' and mode == 'wide-matmul':
        result = np.matmul(inputs[0].astype(np.float64), inputs[1].astype(np.float64))
    elif node.op_type == 'ReduceMean':
        a = attributes(node)
        result = np.mean(inputs[0], axis=tuple(a['axes']) if 'axes' in a else None,
                         keepdims=bool(a.get('keepdims', 1)), dtype=np.float32)
    elif node.op_type == 'Softmax':
        axis = attributes(node).get('axis', -1)
        result = np.exp(inputs[0]-np.max(inputs[0], axis=axis, keepdims=True))
        result /= np.sum(result, axis=axis, keepdims=True, dtype=np.float32)
    else:
        result = execute(node, inputs)
    result = np.asarray(result, dtype=np.float32)
    assert np.isfinite(result).all(), node.name
    return result


def scalar_dots(left, right, output):
    """Independently verify three actual wide-matmul coordinates with fsum."""
    import math
    shape = output.shape
    assert left.ndim >= 2 and right.ndim >= 2 and output.ndim >= 2
    rows = []
    for flat in sorted({0, output.size//2, output.size-1}):
        index = np.unravel_index(flat, shape); batch = index[:-2]
        def operand_batch(value):
            dims = value.shape[:-2]
            tail = batch[len(batch)-len(dims):] if dims else ()
            return tuple(0 if extent == 1 else coordinate for extent, coordinate in zip(dims, tail, strict=True))
        a = left[operand_batch(left)+(index[-2], slice(None))]
        b = right[operand_batch(right)+(slice(None), index[-1])]
        expected = math.fsum(float(x)*float(y) for x, y in zip(a, b, strict=True))
        actual = float(output[index]); error = abs(actual-expected)/max(1., abs(expected))
        assert error <= 1e-7, (index, actual, expected, error)
        rows.append(dict(index=list(map(int, index)), actual=actual, expected=expected, max_scaled=error, reduction=len(a)))
    return rows


def metric(actual, expected, chunked=False):
    assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    a, b = actual.ravel(), expected.ravel()
    if not chunked:
        delta = a.astype(np.float64)-b.astype(np.float64)
        scaled = np.abs(delta)/np.maximum(1., np.abs(b.astype(np.float64)))
        return dict(max_scaled=float(scaled.max()), failed_values=int(np.count_nonzero(scaled > 1e-4)),
                    squared_error=float(np.sum(delta*delta, dtype=np.float64)), values=a.size)
    maximum = 0.; failed = 0; squares = []
    import math
    for start in range(0, a.size, 65537):
        x = np.asarray(a[start:start+65537], dtype=np.float64)
        y = np.asarray(b[start:start+65537], dtype=np.float64)
        differences = x-y
        ratios = np.abs(differences)/np.maximum(np.abs(y), 1.)
        maximum = max(maximum, float(np.max(ratios)))
        failed += int(np.sum(ratios > .0001))
        squares.append(float(np.dot(differences, differences)))
    return dict(max_scaled=maximum, failed_values=failed, squared_error=math.fsum(squares), values=a.size)
