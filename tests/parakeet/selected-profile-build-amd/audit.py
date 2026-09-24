"""Independently reconcile the profile consumer build, its exact method inventory and resource logs."""
import json
from run import BASE, prepared
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import inventory


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json'); payload = read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None and receipt['payload'] == pin(BASE/'payload.json')
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    for name, wanted in read(BASE/'bundle/stage.json')['files'].items():
        assert payload['files'][name] == wanted and pin(BASE/'bundle'/name) == wanted, name
    state = read(collected/'identity.json')
    assert state['complete'] and state['code'] == 0 and state['supervisor'] == read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS and state['boot_time'] == 1789634288.0
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started'] < 4*3600
    assert (collected/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    bridge = read(collected/'bridge-built.json'); assert bridge['passed']
    for name,wanted in bridge['files'].items(): assert pin(collected/name)==wanted,name
    built = read(collected/'built.json'); assert built['passed']
    for name, wanted in built['files'].items():
        if name.startswith('runtime/'): assert pin(collected/name) == wanted, name
    for name, wanted in built['product'].items():
        assert pin(collected/'runtime'/name) == wanted
        assert built['files']['source/bin/Release/net10.0/'+name] == wanted
    assert pin(collected/'runtime/SampledAudio.dll') == built['consumer']
    il = inventory(read(collected/'inventory/instructions.json'),payload['previous_consumer'],built['consumer'],payload['product'])
    assert il == read(collected/'inventory/review.json')
    analysis = dict(passed=True,product=built['product'],consumer=built['consumer'],inventory=il,resources=resources,
        root_product_changed=False,no_profile_capture=True,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    files = {p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__ == '__main__': main()
