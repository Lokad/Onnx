"""Preserve the diagnostic consumer compile failure before any model load."""
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
    assert state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == ['sdk-version', 'census-restore', 'census-build']
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    for row, code in zip(state['runs'], [0, 0, 1], strict=True):
        assert row['complete'] and row['code'] == code
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(s) for s in (collected / 'logs' / (row['name'] + '.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples: assert sample['job'] == row['name']; check_sample(sample)
    assert (collected / 'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    output = (collected / 'logs/census-build.stdout').read_text()
    assert "error CS1061: 'DenseTensor<float>' does not contain a definition for 'Dims'" in output
    assert '3 Error(s)' in output and 'Build FAILED.' in output
    assert not (collected / 'built.json').exists() and not (collected / 'runtime').exists()
    result = dict(passed=False, phase='diagnostic consumer build', model_loads=0, inference_calls=0,
                  reason='Dims is available through ITensor; the adapted consumer used a DenseTensor<float> expression in three places.',
                  repair='Access the same existing Dims property through ITensor in a fresh namespace; all product identities and checks remain unchanged.')
    save(BASE / 'analysis.json', result)
    save(BASE / 'closed.json', dict(**result, files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()},
         local_inputs=spec['files'], remote_terminal=receipt['identities'], generator=pin(__file__)))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), **result)))


if __name__ == '__main__': main()
