"""Independently reconcile every shared/e5 tensor, native reference and VM worker."""
import copy
import json
from run import BASE, prepared
from prepare import ROOT, OLD, FAILED
from scope import census
from protocol import JOBS, RETAINED, ALL_CHECKS, LIMITS, check_sample, pin, read, save
from checks import qualify


def provenance(payload, collected):
    original=read(OLD/'closed.json')['files']
    for name,entry in read(collected/'provenance.json').items():
        if name.startswith(('reference/','e5/')) or name in ['evidence/shared-historical.json','evidence/e5-historical.json']:
            wanted={k:entry[k] for k in ['bytes','sha256']}
            assert pin(collected/name)==wanted==original[entry['source']]
    for name,wanted in payload['model_assets'].items():
        relative=name.removeprefix('/home/vermorel/Onnx/')
        assert original[relative]==wanted==pin(ROOT/relative)


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
    assert payload['retained_failure']==pin(FAILED/'closed.json')
    assert read(collected/'evidence/failed-closed.json')==read(FAILED/'closed.json')
    assert payload['arithmetic_scope']==read(collected/'arithmetic-scope.json')==census(ROOT,read(collected/'reference/manifest.json'))
    for name in RETAINED:
        source=FAILED/'collected'/name/'output'
        assert {p.name:pin(p) for p in source.iterdir()}=={p.name:pin(p) for p in (collected/name/'output').iterdir()}
    results = {}
    for name in ALL_CHECKS:
        results[name] = qualify(collected, name, payload)
        assert results[name] == (read(collected/'retained-reviews.json')[name] if name in RETAINED else read(collected/name/'review.json'))
    for role in ['selected','candidate']:
        assert sum(results[role+'-'+mode]['arrays'] for mode in ['shared','e5'])==166
        assert sum(results[role+'-'+mode]['values'] for mode in ['shared','e5'])==5000814
    analysis = dict(passed=True, identities=payload['identities'], consumer=payload['consumer'], results=results,
        resources=resources, retained_resources=read(FAILED/'analysis.json')['resources'], retained_failure=pin(FAILED/'closed.json'),
        arithmetic_scope=payload['arithmetic_scope'], reused_jobs=RETAINED, executed_jobs=JOBS, reference_provenance_verified=True, no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=spec['files'], remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),passed=True,arrays_per_role=166,values_per_role=5000814,
        maxima={name:max(row['maximum'] for row in value['rows']) for name,value in results.items()},resources=resources)))



if __name__ == '__main__': main()
