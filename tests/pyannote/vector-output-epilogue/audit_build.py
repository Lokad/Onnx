"""Independently retain complete numerical outcomes and exact source boundaries."""
import json
from build import BASE, RAW, MODEL, SCREEN, pin, read, save, verify, monitor
from transform import transform


def main():
    assert not (BASE/'closed.json').exists()
    value = read(BASE/'verified.json'); assert value['complete']; verify(value['files'])
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] in [0, 1]
    raw = read(BASE/'output/raw-256.json')
    assert [r['name'] for r in state['runs']] == ['restore', 'build', 'raw-256'] + (['model-256'] if raw['passed'] else [])
    identities = [state['supervisor']]; resources = []
    for row in state['runs']:
        numerical = row['name'].endswith('-256')
        assert row['complete'] and row['code'] in ([0, 1] if numerical else [0]) and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if numerical else 8)*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            assert sample['seconds'] < 900 and sample['rss'] < 8*1024**3 and sample['available'] >= 1024**3
            assert sample['disk'] >= 20*1024**3 and sample['output_bytes'] <= 1024**3
            assert sample['rss'] == sum(m['rss'] for m in sample['members']) and (not numerical or len(sample['members']) <= 1)
            for member in sample['members']: assert member['affinity'] == [2] and row['members'][str(member['pid'])] == member['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: monitor.terminal(identity)
    for name, wanted in value['reports'].items(): assert pin(BASE/'output'/name) == wanted
    assert raw['cases'] == len(raw['observations']) == 2648 and raw['geometries'] == 312 and raw['layout_cases'] == 331
    assert raw['finite_kernel_cases'] == sum(r['kernel'] for r in raw['observations']) == 2496
    assert raw['nonfinite_fallback_cases'] == 152 and raw['rejected'] == 10 and raw['owned_outputs']
    assert len(raw['supplemental']) == 20 and sum(r['finite'] for r in raw['supplemental']) == 4
    assert all(r['kernel'] == (not r['special']) for r in raw['observations'])
    assert all(r['kernel'] == r['finite'] for r in raw['supplemental'])
    for name in ['scalar_differences', 'production_differences']: assert raw[name] == sum(r[name] for r in raw['observations'])
    assert raw['failed_cases'] == sum(r['scalar_differences'] != 0 or r['production_differences'] != 0 for r in raw['observations']) + sum(r['differences'] != 0 for r in raw['supplemental'])
    assert raw['passed'] == (raw['failed_cases'] == 0)
    original = read(RAW/'output/256.json')
    if raw['passed']: assert raw['observations'] == original['observations'] and raw['supplemental'] == original['supplemental']
    reports = dict(raw=raw)
    if raw['passed']:
        model = read(BASE/'output/model-256.json'); reports['model'] = model
        assert model['cases'] == len(model['observations']) == 108 and model['values'] == 119823360 and model['eligible'] == 96 and model['fallback'] == 12
        assert model['differences'] == sum(r['differences'] for r in model['observations'])
        assert model['native_failures'] == sum(r['native_failed'] for r in model['observations'])
        assert model['failed'] == sum(r['differences'] != 0 or r['native_failed'] != 0 for r in model['observations'])
        assert model['passed'] == (model['failed'] == 0) and model['read_only_operands'] and model['prepared_weights'] == 32
        if model['passed']: assert model['observations'] == read(MODEL/'output/256.json')['observations']
    for mode, result in reports.items():
        row, = [r for r in state['runs'] if r['name'] == mode+'-256']
        assert result['pid'] == row['worker']['pid'] and result['passed'] == (row['code'] == 0)
        assert result['lanes'] == 8 and not result['flags'] and result['core'] == value['core']['sha256'] and result['executable'] == value['consumer']['sha256']
    assert value['passed'] == (state['code'] == 0) == all(r['passed'] for r in reports.values())
    for folder, names in [(RAW, ['GeneratedKernels.cs', 'Probe.cs']), (MODEL, ['ModelProbe.cs']), (SCREEN, ['ComponentBench.cs'])]:
        for name in names: assert pin(BASE/'source'/name) == pin(folder/'source'/name)
    candidate, diff = transform((RAW/'source/BlockedSpatial.cs').read_text())
    assert (BASE/'source/BlockedSpatial.cs').read_text() == candidate and (BASE/'source.diff').read_text() == diff
    summary = {name: {k: r[k] for k in ['passed', 'cases', 'values']} for name, r in reports.items()}
    analysis = dict(passed=value['passed'], qualification_complete=True, reports=summary, resources=resources, no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=value['passed'], qualification_complete=True, files=files, local_inputs=value['files'], identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()
