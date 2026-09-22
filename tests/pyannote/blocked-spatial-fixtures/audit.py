"""Reconcile all captured tensors, graph endpoints, owners and resource samples."""
import collections
import json
import math
from run import BASE, ROOT, pin, read, save, verify, monitor


def main():
    assert not (BASE/'closed.json').exists()
    verified = read(BASE/'verified.json'); assert verified['passed']; verify(verified['files'])
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] == 0
    row, = state['runs']; assert row['name'] == 'capture' and row['complete'] and row['code'] == 0 and row['seconds'] < 900
    assert row['preflight']['available'] >= 12*1024**3
    identities = [state['supervisor']] + [dict(pid=int(p), birth=b) for p, b in row['members'].items()]
    for identity in identities: monitor.terminal(identity)
    samples = [json.loads(s) for s in (BASE/'logs/capture.samples.jsonl').read_text().splitlines()]
    assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
    for sample in samples:
        assert sample['seconds'] < 900 and sample['rss'] < 8*1024**3
        assert sample['available'] >= 1024**3 and sample['disk'] >= 20*1024**3 and sample['output_bytes'] <= 1024**3
        assert sample['rss'] == sum(m['rss'] for m in sample['members']) and len(sample['members']) <= 1
        for member in sample['members']: assert member['affinity'] == [2] and row['members'][str(member['pid'])] == member['birth']
    result = read(BASE/'output/result.json'); assert pin(BASE/'output/result.json') == verified['result'] and result['passed']
    assert result['pid'] == row['worker']['pid'] and result['birth'] == row['worker']['birth'] and result['affinity'] == [2]
    declaration = read(BASE/'output/capture-spec.json'); assert pin(BASE/'output/capture-spec.json') == result['capture_spec']
    assert declaration['derived'] == pin(BASE/'output/layer-capture.onnx') and declaration['original_nodes_and_initializers_unchanged']
    assert len(result['calls']) == 108 and sum(r['eligible'] for r in result['calls']) == 96
    assert len({r['case'] for r in result['calls']}) == 3 and len({r['node'] for r in result['calls']}) == 36
    census = read(ROOT/'artifacts/pyannote-blocked-spatial-census-20260922/census.json')
    for case in {r['case'] for r in result['calls']}:
        rows = [r for r in result['calls'] if r['case'] == case]
        assert [r['index'] for r in rows] == list(range(36))
        assert collections.Counter(r['form'] for r in rows) == {f['index']: f['multiplicity'] for f in census['forms']}
        for r in rows:
            form = census['forms'][r['form']]
            for key in ['input', 'weights', 'output']:
                assert r[key]['shape'] == form[('weight' if key == 'weights' else key)+'_shape']
            assert (r['residual'] is not None) == form['residual'] and r['eligible'] == form['eligible']
    assert len(result['checks']) == 3
    for check in result['checks']:
        assert check['repeated_and_held_exact'] and check['input_unchanged']
        for name in ['unmodified_vs_retained', 'captured_vs_unmodified']: assert check[name]['failed_values'] == 0 and check[name]['maximum'] <= 1e-4
    assert len(result['tensors']) == result['arrays']
    for tensor in result['tensors'].values():
        assert pin(BASE/'output'/tensor['path']) == {k: tensor[k] for k in ['bytes', 'sha256']}
        assert tensor['bytes'] == math.prod(tensor['shape'])*4
    assert sum(t['bytes'] for t in result['tensors'].values()) == result['tensor_bytes'] == declaration['expected_tensor_bytes']
    analysis = dict(passed=True, calls=108, eligible=96, fallback=12, arrays=result['arrays'], tensor_bytes=result['tensor_bytes'],
        resources=len(samples), peak_rss=row['peak_rss'], native_checks=result['checks'], no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=verified['files'], identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()
