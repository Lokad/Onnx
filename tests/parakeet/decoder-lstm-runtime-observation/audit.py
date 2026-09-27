"""Freeze complete diagnostic evidence; attribution is a separate retained-data read."""
import json
from protocol import JOBS, LIMITS, pin, read, save, check_sample
from prepare import BASE, previous_closed
from run import prepared
from checks import result
from events import reconcile


def main():
    assert not (BASE/'closed.json').exists() and not (BASE/'failed.json').exists()
    previous_closed(); prepared()
    folder = BASE/'collected'; receipt = read(folder/'collection.json')
    transfer = read(BASE/'collection-transfer.json'); payload = read(BASE/'payload.json')
    assert receipt['terminal'] and receipt['input_error'] is None
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz')
    assert transfer['receipt'] == pin(folder/'collection.json')
    assert receipt['payload'] == pin(BASE/'payload.json') == pin(folder/'payload.json')
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    state = read(folder/'identity.json')
    assert state['complete'] and state['code'] == receipt['code']
    assert state['supervisor'] == read(BASE/'deployment.json') and state['boot_time'] == 1789634288.0
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p), birth=b) for r in state['runs'] for p,b in r['members'].items()]
    if state['code'] != 0:
        save(BASE/'failed.json', dict(passed=False, terminal=True, state=state,
            files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}))
        raise AssertionError('Original failure retained; diagnose without replay.')
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS
    assert payload['limits'] == LIMITS and state['ended']-state['started'] < 4*3600
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert all(code == 0 for code in row['exitcodes'].values())
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] and samples
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert row['members'][str(member['pid'])] == member['birth']
                assert row['affinities'][str(member['pid'])] == member['expected_affinity']
        assert max(s['rss'] for s in samples) == row['peak_rss']
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    assert set(state['runs'][4]['processes']) == {'worker', 'collector'}
    built = read(folder/'built.json')
    for name, wanted in built['files'].items(): assert pin(folder/name) == wanted, name
    review = result(folder, payload, state['runs'][4], built)
    assert review == read(folder/'trace-capture/review.json')
    events = [json.loads(line) for line in (folder/'trace-export/events/events.jsonl').read_text().splitlines()]
    summary = read(folder/'trace-export/events/summary.json')
    assert summary['input_sha256'] == pin(folder/'trace-capture/capture.nettrace')['sha256']
    value = read(folder/'trace-capture/output/result.json')
    joined = reconcile(value, events, summary)
    save(BASE/'intervals.json', joined)
    stacks = list((folder/'trace-stacks').glob('*.speedscope.json')); assert len(stacks) == 1
    document = read(stacks[0]); assert document['profiles'] and document['shared']['frames']
    phases = [dict(phase=phase, repeat=repeat,
        milliseconds=sum(r['elapsed_ms'] for r in joined['intervals'] if r['phase'] == phase and r['repeat'] == repeat),
        pause_delta_ms=sum(r['pause_delta_ms'] for r in joined['intervals'] if r['phase'] == phase and r['repeat'] == repeat))
        for phase in ['warmup', 'measured'] for repeat in range(5)]
    analysis = dict(passed=True, diagnostic_only=True, root_product_changed=False,
        original_screen_revised=False, product=payload['product'], consumer=built['consumer'],
        result=review, resources=resources, intervals=pin(BASE/'intervals.json'),
        event_summary=summary, clock_offset_bounds_ms=joined['offset_bounds_ms'], phases=phases,
        runtime_attribution_pending=True)
    save(BASE/'analysis.json', analysis)
    save(BASE/'closed.json', dict(passed=True, diagnostic_only=True, performance_admitted=False,
        analysis=pin(BASE/'analysis.json'), files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(passed=True, closed=pin(BASE/'closed.json'), events=len(events), calls=3800,
        clock_offset_bounds_ms=joined['offset_bounds_ms'], phases=phases)))


if __name__ == '__main__': main()
