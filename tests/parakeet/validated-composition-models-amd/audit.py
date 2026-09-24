"""Independently reconcile every Parakeet tensor, public request and VM worker."""
import copy
import json
from run import BASE, prepared
from prepare import PREVIOUS,CURRENT
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import qualify


def provenance(payload, collected):
    original = read(CURRENT/'collected/manifests/candidate-parakeet.json')
    assert read(collected/'evidence/original-manifest.json') == original
    for role in ['selected', 'candidate']:
        expected = copy.deepcopy(original)
        expected.update(core_sha256=payload['identities'][role]['Lokad.Onnx.dll']['sha256'],
                        data_sha256=payload['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
        expected['product_source'] = 'M66 selected release' if role == 'selected' else 'M66 admitted-source composition'
        assert read(collected/'manifests'/(role+'-parakeet.json')) == expected
    original_payload = read(PREVIOUS/'payload.json')
    for name, wanted in payload['files'].items():
        if name.startswith(('assets/', 'parakeet-reference/')):
            assert original_payload['files'][name] == wanted and pin(collected/name) == wanted, name
    for role,products in payload['identities'].items():
        for name,wanted in products.items():assert pin(collected/'runtimes'/role/name)==wanted
        for name,wanted in payload['consumers'].items():assert pin(collected/'runtimes'/role/(name+'.dll'))==wanted
    assert pin(collected/'parakeet-reference/manifest.json')['sha256'] == '3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c'


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json'); payload = read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None and receipt['payload'] == pin(BASE/'payload.json')
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    provenance(payload, collected)
    state = read(collected/'identity.json')
    assert state['complete'] and state['code'] == 0 and state['supervisor'] == read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS and state['boot_time'] == 1789634288.0
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['instruction_environment'] == ({} if row['name'].endswith('-512') else {'DOTNET_EnableAVX512':'0'})
        assert row['preflight'] == row['preflight_observations'][-1]
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert len(sample['members']) <= 1
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        gaps = [samples[0]['seconds']]+[b['seconds']-a['seconds'] for a, b in zip(samples, samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    assert state['ended']-state['started'] < 4*3600
    results = {}
    for name in JOBS:
        results[name] = qualify(collected, name, payload)
        assert results[name] == read(collected/name/'review.json')
    analysis = dict(passed=True, identities=payload['identities'], consumers=payload['consumers'], results=results,
        resources=resources, reference_provenance_verified=True, no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=spec['files'], remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), passed=True, arrays_per_role=784, values_per_role=3090494,
        public_requests_per_role=20, native_maxima={name:results[name]['native']['maximum'] for name in JOBS if '-native-' in name}, resources=resources)))


if __name__ == '__main__': main()
