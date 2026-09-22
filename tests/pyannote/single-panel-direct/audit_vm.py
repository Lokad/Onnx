"""Audit all new AMD component cases, raw blocks, controls and terminal owners."""
import json
import math
import statistics
from vm import ROOT, TOOLS, LOCAL, BASE, PRIOR, common, transport, prepared, pin, read, save, verify, rel


def main():
    assert not (BASE / 'closed.json').exists()
    prep = prepared(); spec = read(BASE / 'payload/payload.json')
    assert spec['gates'] == dict(process_max_min=1.10, geomean_candidate_baseline=.95, max_shape_candidate_baseline=1.05)
    shapes = read(BASE / 'payload/shapes.json')
    assert shapes == read(LOCAL / 'shapes.json') == read(PRIOR / 'payload/shapes.json')
    assert len(shapes['shapes']) == 22 and len(shapes['cases']) == 3266
    collected = BASE / 'collected'; receipt = read(collected / 'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(collected / name) == wanted, name
    assert receipt['payload'] == prep['payload'] == pin(collected / 'payload.json')
    transfer = read(BASE / 'collection-transfer.json')
    assert transfer['archive'] == pin(BASE / 'results.tar.gz') and transfer['receipt'] == pin(collected / 'collection.json')
    state = read(collected / 'identity.json')
    assert state['complete'] and state['code'] == 0 and state['boot_time'] == spec['boot_time']
    assert [r['name'] for r in state['runs']] == spec['jobs']
    identities = [state['supervisor']] + [r['processes']['target'] for r in state['runs']]
    assert receipt['identities'] == [read(collected / 'deployment.json')] + identities[1:]
    live = json.loads(transport.ssh(transport.PRELUDE + f'\nids={identities!r}\nassert not any(live(i) for i in ids)\nprint(json.dumps(dict(terminal=True)))\n'))
    assert live['terminal']
    resources = []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
        assert run['preflight']['available'] >= 8*1024**3 and run['preflight']['tmpfs'] >= 3*1024**3
        samples = [json.loads(line) for line in (collected / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(s['rss'] for s in samples) == run['peak_rss']
        for s in samples:
            assert 0 <= s['seconds'] < 900 and s['rss'] < 2*1024**3
            assert s['available'] >= 1024**3 and s['tmpfs'] >= 1024**3 and s['artifacts'] <= 1024**3
            assert s['rss'] == sum(m['rss'] for m in s['members']) and s['monitor_affinity'] == [0]
            for member in s['members']:
                assert {k: member[k] for k in ['pid', 'birth']} == run['processes']['target']
                assert member['affinity'] == [2] and member['threads'] and all(t['affinity'] == [2] for t in member['threads'])
        resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss'], seconds=run['seconds']))
    results = {name: read(collected / 'output' / (name + '.json')) for name in spec['jobs']}
    for name, value in results.items():
        assert value['passed'] and value['processor_count'] == 1 and value['fma'] and value['avx2'] and value['avx512']
        assert value['flags'] == [] and value['runtime'] == '10.0.8' and value['core'] == common.CORE
        assert value['executable'] == spec['scalar_consumer' if name == 'validate-scalar-tail' else 'consumer']['sha256']
        assert value['shapes'] == pin(BASE / 'payload/shapes.json')['sha256']
        assert value['mode'] == name.split('-')[0]
        assert value['pid'] == next(r['processes']['target']['pid'] for r in state['runs'] if r['name'] == name)
        if name.startswith('validate'):
            assert [{k: r[k] for k in ['m', 'n', 'k', 'stride', 'start', 'bias', 'pattern']} for r in value['records']] == shapes['cases']
            assert all(r['passed'] and r['values'] == r['m']*r['k'] and r['checked_buffer_values'] == r['m']*r['stride']+6 for r in value['records'])
            assert value['conditioning'] == []
        else:
            assert len(value['records']) == 132
            assert [(r['m'],r['n'],r['k'],r['stride'],r['start']) for r in value['conditioning']] == [(s['m'],s['n'],s['k'],s['stride'],s['timing_start']) for s in shapes['shapes']]
            assert all(r['calls'] >= 16 and r['seconds'] >= 1 for r in value['conditioning'])
    normal = results['validate']['records']
    for value in [results['validate-scalar-tail'], read(LOCAL / 'output/validate-normal.json'), read(LOCAL / 'output/validate-scalar.json')]:
        assert [{k:v for k,v in r.items() if k != 'digest'} for r in value['records']] == [{k:v for k,v in r.items() if k != 'digest'} for r in normal]
        assert [r['digest'] for r in value['records'] if r['pattern'] != 'special'] == [r['digest'] for r in normal if r['pattern'] != 'special']
    digests = {(r['m'],r['n'],r['k'],r['stride'],r['start']):r['digest'] for r in normal if r['bias'] and r['pattern'] == 'finite'}
    rows = []
    for shape in shapes['shapes']:
        key = tuple(shape[k] for k in ['m','n','k','stride','timing_start']); means = {}
        for name in spec['jobs'][2:]:
            subset = [r for r in results[name]['records'] if tuple(r[k] for k in ['m','n','k','stride','start']) == key]
            assert [r['block'] for r in subset] == list(range(6))
            for r in subset:
                assert r['iterations'] == shape['iterations'] and r['warm_calls'] >= 16 and r['warm_seconds'] >= 1
                assert r['digest'] == digests[key] and r['seconds'] > 0 and 0 <= r['cpu_seconds'] <= r['seconds']+.1
            means[name] = statistics.fmean(r['seconds']/r['iterations'] for r in subset)
        controls = {role: max(means[role+'-a'],means[role+'-b'])/min(means[role+'-a'],means[role+'-b']) for role in ['baseline','candidate']}
        ratio = (means['candidate-a']+means['candidate-b'])/(means['baseline-a']+means['baseline-b'])
        rows.append(dict(**shape, process_means=means, controls=controls, controls_passed=all(v <= 1.10 for v in controls.values()), candidate_baseline=ratio))
    geo = math.exp(statistics.fmean(math.log(r['candidate_baseline']) for r in rows))
    worst = max(r['candidate_baseline'] for r in rows); controls = all(r['controls_passed'] for r in rows)
    analysis = dict(passed=True, eligible=controls and geo <= .95 and worst <= 1.05, controls_passed=controls,
        geomean_candidate_baseline=geo, max_shape_candidate_baseline=worst, gates=spec['gates'], rows=rows,
        validation_cases=3266, validation_modes=4, measured_blocks=528, resources=resources,
        remote_identities=identities, local_closure=pin(LOCAL / 'closed.json'),
        scope='Combined direct-output and narrow packing omission versus production component; no isolated causal copy cost or application/ORT timing.')
    save(BASE / 'analysis.json', analysis)
    files = dict(prep['files'])
    files.update({rel(p):pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE / 'closed.json', dict(passed=True, files=files, remote_identities=identities, analysis=pin(BASE / 'analysis.json')))
    print(json.dumps(dict(eligible=analysis['eligible'], controls=controls, geomean=geo, worst=worst,
         resource_samples=sum(r['samples'] for r in resources), closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__': main()
