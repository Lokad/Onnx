"""Close the missing-global.json refusal before any test restore, build or test."""
import json
from run import BASE, prepared
from protocol import LIMITS, check_sample, pin, read, save


def main():
    spec = prepared(); assert not (BASE / 'closed.json').exists()
    collected = BASE / 'collected'; receipt = read(collected / 'collection.json')
    transfer = read(BASE / 'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE / 'results.tar.gz')
    assert transfer['receipt'] == pin(collected / 'collection.json')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['input_error'] is None
    assert receipt['payload'] == pin(BASE / 'payload.json')
    for name, wanted in receipt['files'].items(): assert pin(collected / name) == wanted, name
    state = read(collected / 'identity.json')
    assert state['complete'] and state['code'] == 1 and state['supervisor'] == read(BASE / 'deployment.json')
    assert state['boot_time'] == 1789634288.0 and len(state['runs']) == 1
    row = state['runs'][0]
    assert row['name'] == 'sdk-version' and row['complete'] and row['code'] == 0
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for p, b in row['members'].items()]
    assert row['preflight']['available'] >= LIMITS['preflight_available']
    assert row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
    samples = [json.loads(s) for s in (collected / 'logs/sdk-version.jsonl').read_text().splitlines()]
    assert len(samples) == row['samples'] == 1
    check_sample(samples[0]); assert max(s['rss'] for s in samples) == row['peak_rss']
    assert (collected / 'logs/sdk-version.stdout').read_text().strip() == '10.0.300-preview.0.26177.108'
    assert "endswith('10.0.204')" in row['error'] and 'AssertionError' in state['error']
    assert not any((collected / name).exists() for name in ['tests-restore', 'tests-build', 'selected-negative', 'candidate-tests', 'candidate-tests-256', 'built.json'])
    assert not (BASE / 'bundle/source/global.json').exists()
    result = dict(passed=False, expected_infrastructure_refusal=True,
                  reason='The isolated test bundle omitted the repository global.json; SDK guard refused the preview before restore or tests.',
                  selected_sdk='10.0.300-preview.0.26177.108', required_sdk='10.0.204',
                  product_rebuilt=False, tests_run=0, performance_calls=0,
                  repair='Copy the unchanged selected-source global.json into a fresh versioned namespace; preserve every guard.')
    save(BASE / 'analysis.json', result)
    save(BASE / 'closed.json', dict(**result, files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()},
         local_inputs=spec['files'], remote_terminal=receipt['identities'], generator=pin(__file__)))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), **result)))


if __name__ == '__main__': main()
