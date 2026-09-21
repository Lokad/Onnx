"""Fixed whole-layer reference experiment; reuse the qualified reference engines."""
from pathlib import Path
import sys
import os

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_TOOLS = ROOT/'tests/whisper/full-reference'
sys.path.insert(0, str(REFERENCE_TOOLS))
import common as reference
from common import np, pin, read, write, rel, raw, tensor_array, packages, THREADS, absent
packages()
import psutil

BASE = ROOT/'artifacts/whisper-layer20-reference-v2-20260921'
PRIOR = ROOT/'artifacts/whisper-layer20-cross-20260920'
FULL = ROOT/'artifacts/whisper-full-reference-20260920'
MODEL = ROOT/'models/whisper-large-v3-turbo/onnx/encoder_layer20_20260918.onnx'
LIMITS = dict(seconds=900, rss=4*1024**3, available=1024**3, preflight_available=6*1024**3, disk=12*1024**3)
PROTOCOL = 'whisper-layer20-reference-decomposition-v1'


def load(desc):
    path = ROOT/desc['file']; assert pin(path) == desc['pin']
    assert desc['dtype'] in ('<f4', '<f8')
    value = np.fromfile(path, dtype=desc['dtype']).reshape(desc['shape'])
    assert np.isfinite(value).all()
    return value.astype(np.float64)


def metric(delta, denominator):
    assert delta.shape == denominator.shape and np.isfinite(delta).all() and np.all(denominator >= 1)
    a = np.abs(delta); s = a/denominator; i = int(s.argmax())
    square = float(np.sum(delta*delta, dtype=np.float64))
    return dict(max_scaled=float(s.flat[i]), max_absolute=float(a.max()), failed_values=int(np.count_nonzero(s > 1e-4)),
                scaled_index=i, values=delta.size, rms=float(np.sqrt(square/delta.size)), l2=float(np.sqrt(square)))


def decompose(actual, own, ideal):
    assert actual.shape == own.shape == ideal.shape
    total, inherited, local = actual-ideal, own-ideal, actual-own
    closure = total-inherited-local
    bound = 16*np.finfo(np.float64).eps*(np.abs(actual)+np.abs(own)+np.abs(ideal)+1)
    assert np.all(np.abs(closure) <= bound)
    denominator = np.maximum(1, np.abs(ideal))
    return dict(total=metric(total, denominator), inherited=metric(inherited, denominator), local=metric(local, denominator),
                local_own=metric(local, np.maximum(1, np.abs(own))), closure_max=float(np.abs(closure).max()),
                signed_cross_term=float(2*np.sum(inherited*local, dtype=np.float64)))


def check_pair(spec, request, incoming):
    jobs = [j for j in spec['jobs'] if j['request'] == request and j['incoming'] == incoming]
    assert [j['engine'] for j in jobs] == ['numpy', 'ort']
    result = [read(BASE/'outputs'/j['id']/'result.json') for j in jobs]
    maximum = 0.
    for i, desc in enumerate(spec['outputs']):
        arrays = []
        for job, value in zip(jobs, result, strict=True):
            row = value['outputs'][i]; assert row['name'] == desc['name'] and row['shape'] == desc['shape']
            path = BASE/'outputs'/job['id']/row['file']; assert pin(path) == row['pin']
            arrays.append(np.fromfile(path, dtype='<f8').reshape(desc['shape']))
        error = float((np.abs(arrays[0]-arrays[1])/np.maximum(1, np.abs(arrays[0]))).max())
        assert error <= 1e-9, (jobs, i, error)
        maximum = max(maximum, error)
        if incoming == 'reference' and i == 11:
            for full in spec['requests'][request]['full_final'].values():
                expected = load(full)
                for actual in arrays:
                    assert float((np.abs(actual-expected)/np.maximum(1, np.abs(expected))).max()) <= 1e-9
    return maximum
