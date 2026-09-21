"""Read xUnit's individual skipped outcomes and preserve its raw TRX counters."""
import collections
import xml.etree.ElementTree as ET
from common import *


def read_suite(base, name, passed, skipped):
    path = base / 'test-results' / (name + '.trx')
    tree = ET.parse(path)
    counters = tree.find('.//{*}Counters').attrib
    outcomes = collections.Counter(r.attrib['outcome'] for r in tree.findall('.//{*}UnitTestResult'))
    assert outcomes == dict(Passed=passed, **({'NotExecuted': skipped} if skipped else {})), outcomes
    assert int(counters['passed']) == int(counters['executed']) == passed
    assert int(counters['total']) == passed + skipped and int(counters['failed']) == 0
    return dict(name=name, counters=counters, outcomes=dict(outcomes), trx=pin(path))


def resources(base, names, code):
    state = read(base / 'processes.json')
    assert state['complete'] and state['code'] == code
    assert [r['name'] for r in state['runs']] == names
    identities, rows = [state['supervisor']], []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
        testing = run['name'] in ['focused', 'hardware-disabled', 'backend-full', 'tensors-full', 'consumer']
        assert run['preflight']['available'] >= (10 if testing else 8) * 1024**3
        samples = [json.loads(line) for line in (base / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(r['rss'] for r in samples) == run['peak_rss']
        for row in samples:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3
            assert row['disk'] >= 20 * 1024**3 and row['output_bytes'] <= 1024**3
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
            assert run['name'] != 'consumer' or len(row['members']) <= 1
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        rows.append(dict(name=run['name'], seconds=run['seconds'], samples=len(samples), peak_rss=run['peak_rss']))
    for identity in identities:
        terminal(identity)
    return dict(identities=identities, resources=rows, resource_samples=sum(r['samples'] for r in rows),
        peak_rss=max(r['peak_rss'] for r in rows))
