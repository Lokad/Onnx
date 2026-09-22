"""Audit normal source preparation; model/application qualification stays open."""
import json
import sys
import xml.etree.ElementTree as ET
from prepare import BASE, pin, read, save, verify, terminal, review


def main():
    assert sys.argv[1:] == ['prepare'] and not (BASE/'preparation-closed.json').exists()
    proof = read(BASE/'prepared.json'); assert proof['passed']; verify(proof['files'])
    assert not proof['production_changed'] and not proof['models_qualified'] and not proof['performance_qualified']
    assert review() == read(BASE/'instruction-review.json')
    state = read(BASE/'preparation.json'); assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == [n+'-'+phase for n in ['cli', 'backend', 'tensors', 'bridge'] for phase in ['restore', 'build']] + ['inventory', 'focused-normal', 'focused-disabled']
    resources = []; identities = [state['supervisor']]
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if row['name'].startswith('focused-') else 8)*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            assert sample['seconds'] < 900 and sample['rss'] < 8*1024**3 and sample['available'] >= 1024**3
            assert sample['disk'] >= 20*1024**3 and sample['output_bytes'] <= 1024**3
            assert sample['rss'] == sum(m['rss'] for m in sample['members'])
            for member in sample['members']: assert member['affinity'] == [2] and row['members'][str(member['pid'])] == member['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: terminal(identity)
    suites = []
    ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    for mode in ['normal', 'disabled']:
        path = BASE/'test-results'/('focused-'+mode+'.trx'); tree = ET.parse(path)
        counters = tree.find('.//t:Counters', ns).attrib
        rows = tree.findall('.//t:UnitTestResult', ns)
        assert int(counters['failed']) == 0 and int(counters['passed']) == int(counters['total']) == len(rows) == 31
        assert all(r.attrib['outcome'] == 'Passed' for r in rows)
        suites.append(dict(mode=mode, tests=len(rows), names=[r.attrib['testName'] for r in rows], file=pin(path)))
    analysis = dict(passed=True, suites=suites, resources=resources, source_review=read(BASE/'instruction-review.json'),
        models_qualified=False, performance_qualified=False, production_changed=False)
    save(BASE/'preparation-analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'preparation-closed.json', dict(passed=True, files=files, local_inputs=proof['files'], identities=identities, analysis=pin(BASE/'preparation-analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'preparation-closed.json'), suites=[dict(mode=s['mode'], tests=s['tests']) for s in suites],
        samples=sum(r['samples'] for r in resources), changed=analysis['source_review']['observations'][0]['changed'])))


if __name__ == '__main__': main()
