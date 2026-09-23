"""Independently reconcile every Parakeet request, exact clock, retained gate and VM worker."""
import copy
import json
from run import BASE, prepared
from prepare import APP
from protocol import JOBS, TIMING_ROLES, LIMITS, check_sample, pin, read, save
from checks import qualify, prereqs, evaluate
from statistics_exact import timing_table
import importlib.util


def provenance(payload, collected):
    original = read(APP/'payload.json')
    manifest = APP/'collected/manifests/candidate-parakeet.json'
    assert pin(collected/'manifests/current-parakeet.json') == pin(manifest) == pin(collected/'evidence/original-parakeet.json')
    for name, wanted in payload['files'].items():
        if name.startswith(('assets/', 'runtime/', 'runtimes/')):
            source = name.replace('runtimes/current/', 'runtimes/candidate/')
            assert original['files'][source] == wanted, name
            if not name.startswith('assets/'): assert pin(collected/name) == wanted, name
    assert payload['external'] == original['external']
    assert payload['interpreter'] == original['interpreter'] and payload['python_paths'] == original['python_paths']
    for name, wanted in payload['identities']['current'].items(): assert pin(collected/'runtimes/current'/name) == wanted
    for name, wanted in payload['consumers'].items(): assert pin(collected/'runtimes/current'/(name+'.dll')) == wanted


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
    module=importlib.util.spec_from_file_location('accounting_audit',collected/'runtime/campaign_processes.py')
    account=importlib.util.module_from_spec(module);module.loader.exec_module(account)
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['preflight'] == row['preflight_observations'][-1]
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        gaps = [samples[0]['seconds']]+[b['seconds']-a['seconds'] for a, b in zip(samples, samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0 <= gap < 10 for gap in gaps)
        accounting=account.foreign_fraction(read(collected/row['name']/'pre.json'),read(collected/row['name']/'post.json'),state['supervisor']['pid'])
        assert accounting==row['accounting'] and accounting['valid']
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds'],accounting=accounting))
    assert state['ended']-state['started'] < 4*3600
    results = {}
    for name in JOBS:
        results[name] = qualify(collected, name, payload)
        assert results[name] == read(collected/name/'review.json')
    prior=prereqs(collected,payload)
    timing=[read(collected/f'timing-{i:02}-{role}'/'output/result.json') for i,role in enumerate(TIMING_ROLES)]
    assert sum(len(value['records']) for value in timing)==320
    assert sum(r['phase']=='warmup' for value in timing for r in value['records'])==80
    assert sum(r['phase']=='measured' for value in timing for r in value['records'])==240
    for row in state['runs']:
        if row['name'].startswith(('native-','timing-')):
            value=read(collected/row['name']/'output/result.json')
            assert value['setup_seconds']+sum(r['seconds'] for r in value['records'])<=row['seconds']
    table=timing_table(timing,read(collected/'manifests/current-parakeet.json'))
    performance=evaluate(table)
    analysis=dict(passed=True,identities=payload['identities'],consumers=payload['consumers'],prerequisites=prior,
        results=results,resources=resources,table=table,performance=performance,
        no_new_conformance_campaign=True,timing_requests=320,warmup=80,measured=240,
        reference_provenance_verified=True,application_parity_target=1.05)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=spec['files'], remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),passed=True,performance=performance,table=table,
        resource_samples=sum(r['samples'] for r in resources),peak_rss=max(r['peak_rss'] for r in resources))))



if __name__ == '__main__': main()
