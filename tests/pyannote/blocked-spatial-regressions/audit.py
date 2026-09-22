"""Independently reconcile test outcomes, bytes and resource observations."""
import json
from run import BASE, SUITES, pin, read, save, verify, terminal, suite, prior


def main():
    assert not (BASE/'closed.json').exists()
    value = read(BASE/'verified.json'); assert value['passed']; verify(value['files']); prior()
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == [r[0] for r in SUITES]
    outcomes = [suite(name, passed, skipped) for name, _, passed, skipped in SUITES]
    assert outcomes == value['suites'] == read(BASE/'suites.json')
    identities = [state['supervisor']]; resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < 900
        assert row['preflight']['available'] >= 12*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for s in samples:
            assert s['seconds'] < 900 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
            assert s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3 and s['rss'] == sum(m['rss'] for m in s['members'])
            for m in s['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: terminal(identity)
    result = dict(passed=True, suites=[dict(name=r['name'], passed=r['passed'], skipped=r['skipped']) for r in outcomes],
        corrected_focused_tests_separately=31, single_combined_suite_not_yet_run=True,
        core=value['core'], data=value['data'], resources=resources, no_performance_measurement=True)
    save(BASE/'analysis.json', result)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=value['files'], identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **result)))


if __name__ == '__main__': main()
