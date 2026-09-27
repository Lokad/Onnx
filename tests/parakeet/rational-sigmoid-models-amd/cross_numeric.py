"""Compare all saved model outputs without requiring approximate floats to match bits."""
import hashlib
import math
import numpy as np
from protocol import read


def compare_values(left, right, dtype, shape):
    types = {'Float': np.dtype('<f4'), 'Int32': np.dtype('<i4'), 'Int64': np.dtype('<i8')}
    actual_type = types[dtype]
    assert all(type(d) is int and d >= 0 for d in shape)
    count = math.prod(shape)
    assert len(left) == len(right) == count * actual_type.itemsize
    a = np.frombuffer(left, dtype=actual_type)
    b = np.frombuffer(right, dtype=actual_type)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    maximum = 0.; index = 0
    if dtype == 'Float':
        delta = np.abs(b.astype(np.float64)-a.astype(np.float64))/np.maximum(1, np.abs(a.astype(np.float64)))
        maximum = float(delta.max()) if count else 0.
        index = int(delta.argmax()) if maximum else 0
        assert maximum <= 1e-4, (maximum, index)
    else:
        assert left == right, 'Integer model output changed'
    return dict(values=count, dtype=dtype, shape=shape, maximum_scaled_error=maximum,
        worst_index=index, bit_identical=left == right, selected_sha256=hashlib.sha256(left).hexdigest(),
        candidate_sha256=hashlib.sha256(right).hexdigest())


def compare_native(base, isa):
    before = read(base/f'selected-native-{isa}/result.json')
    after = read(base/f'candidate-native-{isa}/result.json')
    comparisons = []
    seen = set()
    for a, b in zip(before['rows'], after['rows'], strict=True):
        assert a['name'] == b['name'] and a['actual'] == b['actual']
        for x, y in zip(a['comparisons'], b['comparisons'], strict=True):
            assert all(x[k] == y[k] for k in ['label', 'output', 'shape', 'dtype', 'file'])
            key = (a['name'], x['label'], x['output'])
            assert key not in seen; seen.add(key)
            files = []
            for role, entry in [('selected', x), ('candidate', y)]:
                root = (base/f'{role}-native-{isa}/result.json.tensors').resolve()
                path = (root/entry['file']).resolve()
                assert path.is_relative_to(root) and path != root
                data = path.read_bytes(); assert hashlib.sha256(data).hexdigest() == entry['sha256']
                files.append(data)
            comparisons.append(dict(case=a['name'], label=x['label'], output=x['output'],
                **compare_values(*files, x['dtype'], x['shape'])))
    assert len(comparisons) == 784 and sum(r['values'] for r in comparisons) == 3090494
    return comparisons
