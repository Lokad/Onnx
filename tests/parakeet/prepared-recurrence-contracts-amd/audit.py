"""Independently reconcile focused packing contracts and the expected negative control."""
import json
from run import BASE, prepared
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import contract


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json'); payload = read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None and receipt['payload'] == pin(BASE/'payload.json')
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    for name, wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert payload['files'][name] == wanted and pin(BASE/'bundle'/name) == wanted, name
    for name,wanted in payload['files'].items():
        if name.startswith(('measured/','bridge/')):assert pin(collected/name)==wanted,name
    state = read(collected/'identity.json')
    assert state['complete'] and state['code'] == 0 and state['supervisor'] == read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS and state['boot_time'] == 1789634288.0
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == (1 if row['name']=='selected-negative' else 0) and row['seconds'] < LIMITS['seconds']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started'] < 4*3600
    assert (collected/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    built = read(collected/'built.json'); assert built['passed']
    for name, wanted in built['files'].items():
        if name.startswith('runtime/'): assert pin(collected/name) == wanted, name
    reviews = {}
    assert built['identities'] == payload['identities']
    for role, identities in payload['identities'].items():
        for name, wanted in identities.items(): assert pin(collected/'runtime'/role/name) == wanted
        assert pin(collected/'runtime'/role/'Lokad.Onnx.Backend.Tests.dll') == built['consumer']
    for row in state['runs']:
        if row['name'] in ['selected-negative','candidate-tests','candidate-tests-256']:
            value = contract(collected/row['name'], payload, built, row)
            assert value == read(collected/row['name']/'review.json')
            reviews[row['name']] = value
    assert len(reviews) == 3
    analysis = dict(passed=True, identities=payload['identities'], consumer=built['consumer'],
        contracts=reviews, resources=resources, candidate_passes=2*sum(payload['expected_cases'].values()), required_selected_failure=True,
        source_prepared=pin(collected/'evidence/source-prepared.json'),
        product_rebuilt=False, root_product_changed=False, model_qualified=False, no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    files = {p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__ == '__main__': main()
