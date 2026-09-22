"""Close the explicit review/test successor without changing product binaries."""
import json
from run import BASE, PRIOR, pin, read, save, verify, terminal, review, suites


def main():
    assert not (BASE/'closed.json').exists()
    value = read(BASE/'verified.json'); assert value['passed']; verify(value['files'])
    prior = read(PRIOR/'failure-closed.json'); assert prior['retained_failure'] and not prior['passed']
    for name, wanted in prior['files'].items(): assert pin(PRIOR/name) == wanted, name
    assert value['core'] == prior['core'] and value['data'] == prior['data'] and value['product_bytes_unchanged']
    assert review() == read(BASE/'instruction-review.json') and suites() == value['suites']
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == ['restore', 'build', 'focused-normal', 'focused-disabled']
    resources = []; identities = [state['supervisor']]
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if row['name'].startswith('focused-') else 8)*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for s in samples:
            assert s['seconds'] < 900 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
            assert s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3
            assert s['rss'] == sum(m['rss'] for m in s['members'])
            for m in s['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in prior['identities']+identities: terminal(identity)
    test = read(BASE/'selftest.json'); assert test['passed'] and test['tests'] == 7
    analysis = dict(passed=True, preparation_only=True, core=value['core'], data=value['data'],
        product_bytes_unchanged=True, prior_failure=pin(PRIOR/'failure-closed.json'), source_review=review(),
        suites=suites(), resources=resources, original_build_resources=read(PRIOR/'failure-analysis.json')['resources'],
        models_qualified=False, performance_qualified=False)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=True, preparation_only=True, files=files, local_inputs=value['files'],
        identities=identities, prior_identities=prior['identities'], source_artifact=str(PRIOR), analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), suites=[dict(mode=s['mode'], tests=s['tests']) for s in value['suites']],
        unchanged_core=3103, changed_core=10, added_core=48, unchanged_data=697, equivalent_component_methods=11,
        samples=sum(r['samples'] for r in resources))))


if __name__ == '__main__': main()
