"""Independent chunked recomputation from all actual, own-input and ideal arrays."""
import math
import json
import time
from protocol import *


def independently(row):
    arrays = []
    for key in ['actual', 'own', 'ideal']:
        desc = row[key]; path = ROOT/desc['file']; assert pin(path) == desc['pin']
        arrays.append(np.memmap(path, mode='r', dtype=desc['dtype']))
    size = arrays[0].size; assert all(a.size == size for a in arrays)
    accum = {k: dict(max_scaled=-1., max_absolute=0., failed_values=0, scaled_index=0, squares=[]) for k in ['total', 'inherited', 'local', 'local_own']}
    cross, closures = [], []
    for offset in range(0, size, 65536):
        a, b, c = [np.asarray(v[offset:offset+65536], dtype=np.float64) for v in arrays]
        assert all(np.isfinite(v).all() for v in [a, b, c])
        t, inherited, local = a-c, b-c, a-b
        closure = t-(inherited+local)
        assert np.all(np.abs(closure) <= 16*np.finfo(np.float64).eps*(np.abs(a)+np.abs(b)+np.abs(c)+1))
        closures.append(float(np.abs(closure).max())); cross.append(float(np.sum(2*inherited*local)))
        for key, d in [('total', t), ('inherited', inherited), ('local', local), ('local_own', local)]:
            denominator = np.where(np.abs(b if key == 'local_own' else c) > 1, np.abs(b if key == 'local_own' else c), 1.)
            absolute = np.abs(d); scaled = absolute/denominator; i = int(np.argmax(scaled)); item = accum[key]
            if float(scaled[i]) > item['max_scaled']:
                item.update(max_scaled=float(scaled[i]), scaled_index=offset+i)
            item['max_absolute'] = max(item['max_absolute'], float(absolute.max()))
            item['failed_values'] += int(np.sum(scaled > .0001)); item['squares'].append(float(np.dot(d, d)))
    answer = {}
    for key, item in accum.items():
        square = math.fsum(item.pop('squares')); item.update(values=size, rms=math.sqrt(square/size), l2=math.sqrt(square)); answer[key] = item
    answer['signed_cross_term'] = math.fsum(cross)
    # The two algebraic evaluation orders can differ in low double bits.
    answer['closure_max'] = max(closures)
    return answer


def main():
    target = BASE/'verification.json'; assert not target.exists()
    analysis = read(BASE/'audit.json'); assert analysis['passed'] is True and len(analysis['rows']) == 768
    assert all(absent(b) for b in analysis['births'])
    expected = {(r, cell, engine, index) for r in range(8) for cell in ['MM', 'MN', 'NM', 'NN']
                for engine in ['numpy', 'ort'] for index in range(12)}
    assert {(r['request'], r['cell'], r['reference'], r['index']) for r in analysis['rows']} == expected
    checked = 0; started = time.monotonic(); process = psutil.Process(); process.cpu_affinity([2]); resources = []
    for row in analysis['rows']:
        actual = independently(row); wanted = row['metrics']
        for key in ['total', 'inherited', 'local', 'local_own']:
            for name in ['values', 'failed_values', 'scaled_index']:
                assert actual[key][name] == wanted[key][name]
            for name in ['max_scaled', 'max_absolute', 'rms', 'l2']:
                assert abs(actual[key][name]-wanted[key][name]) <= max(1e-15, abs(wanted[key][name])*2e-12)
                checked += 1
        assert abs(actual['signed_cross_term']-wanted['signed_cross_term']) <= max(1e-14, abs(wanted['signed_cross_term'])*1e-10)
        resource = dict(seconds=time.monotonic()-started, rss=process.memory_info().rss, available=psutil.virtual_memory().available)
        assert resource['seconds'] < 1800 and resource['rss'] < LIMITS['rss'] and resource['available'] >= LIMITS['available']
        resources.append(resource)
    write(target, dict(passed=True, audit=pin(BASE/'audit.json'), rows=768, checked_scalars=checked,
                       terminal_births=len(analysis['births']), resources=resources))
    print(json.dumps(dict(passed=True, rows=768, checked_scalars=checked, verification=pin(target))))


if __name__ == '__main__':
    main()
