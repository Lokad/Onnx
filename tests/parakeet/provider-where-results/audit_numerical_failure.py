"""Close the stopped numerical lane without admitting candidate correctness."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/parakeet/provider-where-numerics'))
from run import BASE, prepared
from protocol import JOBS, LIMITS, check_sample, pin, read, save


def main():
    assert not (BASE / 'closed.json').exists()
    prepared()
    c = BASE / 'collected'
    receipt = read(c / 'collection.json')
    state = read(c / 'identity.json')
    transfer = read(BASE / 'collection-transfer.json')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['input_error'] is None
    assert receipt['payload'] == pin(BASE / 'payload.json') and transfer['passed']
    assert transfer['archive'] == pin(BASE / 'results.tar.gz') and transfer['receipt'] == pin(c / 'collection.json')
    for name, wanted in receipt['files'].items():
        assert pin(c / name) == wanted, name
    assert state['complete'] and state['code'] == 1 and state['supervisor'] == read(BASE / 'deployment.json')
    assert state['boot_time'] == 1789634288.0 and state['ended'] - state['started'] < 4 * 3600
    assert [r['name'] for r in state['runs']] == JOBS[:4]
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    resources = []
    for index, row in enumerate(state['runs']):
        assert row['complete'] and row['code'] == (0 if index < 3 else -6) and row['seconds'] < 900
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(s) for s in (c / 'logs' / (row['name'] + '.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        gaps = [samples[0]['seconds']] + [b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])] + [row['seconds']-samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    stderr = (c / 'logs/current-numerics-256.stderr').read_text()
    assert 'System.IO.InvalidDataException: capture-0 profile validation stages' in stderr
    assert not list(c.glob('*/result.json'))
    built = read(c / 'built.json')
    assert built['passed']
    for name, wanted in built['files'].items():
        assert pin(c / name) == wanted
    # Preserve the source basis for correcting the checker. The failed worker
    # did not serialize its stage count; do not invent a measured value.
    names = ['src/Lokad.Onnx/TensorOps.Elementwise.cs', 'src/Lokad.Onnx/TensorOps.Broadcast.cs']
    source = {name: pin(ROOT / name) for name in names}
    parent = read(ROOT / 'artifacts/parakeet-provider-where-source-20260924/prepared.json')['before']
    assert all(value == parent[name] for name,value in source.items())
    elementwise = (ROOT / names[0]).read_text().split('public static Tensor<T> Where(',1)[1].split('public static Tensor<byte> Add(',1)[0]
    assert elementwise.count('StartOpStage(OpStage.ValidateArguments);') == 1
    assert all(s in elementwise for s in ['Broadcast(bx, shape, out var fx)', 'Broadcast(by, shape, out var fy)', 'Tensor<bool>.BroadcastTo(condition, shape)'])
    broadcast = (ROOT / names[1]).read_text()
    wrapper = broadcast.split('public static bool Broadcast(Tensor<T> x, ReadOnlySpan<int> y,',1)[1].split('public static bool BroadcastShape(',1)[0]
    assert wrapper.count('bx = BroadcastTo(x, shape);') == 1
    body = broadcast.split('public static Tensor<T> BroadcastTo(',1)[1].split('public static Tensor<T> Expand(',1)[0]
    assert body.count('StartOpStage(OpStage.ValidateArguments);') == 1
    assert body.count('StartOpStage(OpStage.CalculateIndices);') == 1
    analysis = dict(passed=False, classification='Checker omitted three nested BroadcastTo validation stages.',
        observed_failure='capture-0 profile validation stages', observed_stage_count=None,
        source_predicted_regular_validation_stages=4, source=source,
        completed_jobs=JOBS[:3], stopped_job=JOBS[3], unexecuted_jobs=JOBS[4:], resources=resources,
        candidate_numerics_executed=False, numerical_qualification_admitted=False, root_product_changed=False)
    save(BASE / 'failure-analysis.json', analysis)
    save(BASE / 'closed.json', dict(passed=False, numerical_qualification_admitted=False,
        verifier=pin(__file__), analysis=pin(BASE / 'failure-analysis.json'),
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), **analysis)))


if __name__ == '__main__':
    main()
