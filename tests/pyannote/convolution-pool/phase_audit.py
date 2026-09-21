"""Audit closed owned-process phases without modifying their evidence."""
import json
import xml.etree.ElementTree as ET


def audit_preparation(base, common):
    prepared = common.read(base / 'prepared.json')
    assert prepared['passed']
    common.verify(prepared['files'])
    state = common.read(base / 'preparation.json')
    assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == ['bridge-restore', 'bridge-build', 'instructions', 'backend-restore', 'backend-build', 'focused', 'hardware-disabled']
    common.terminal(state['supervisor'])
    identities = [state['supervisor']]
    samples = 0
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0
        preflight = 10 if run['name'] in ['focused', 'hardware-disabled'] else 8
        assert run['preflight']['available'] >= preflight * 1024**3
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            common.terminal(identity)
            identities.append(identity)
        rows = [json.loads(line) for line in (base / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(r['rss'] for r in rows) == run['peak_rss']
        for row in rows:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3
            assert row['output_bytes'] <= 1024**3 and row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        samples += len(rows)
    suites = []
    for name in ['focused', 'hardware-disabled']:
        tree = ET.parse(base / 'test-results' / (name + '.trx'))
        counters = tree.find('.//{*}Counters').attrib
        assert counters['failed'] == '0' and int(counters['passed']) >= 31
        assert counters == common.read(base / (name + '.json'))['counters']
        suites.append(dict(name=name, counters=counters))
    return dict(passed=True, identities=identities, resource_samples=samples, suites=suites)
