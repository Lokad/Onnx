"""Close the incomplete synthetic graph fixture without discarding its checks."""
import json
import xml.etree.ElementTree as ET
from common import ROOT, TOOLS, pin, read, save, verify, terminal, rel

BASE = ROOT / 'artifacts/pyannote-convolution-pool-v3-20260921'


def main():
    assert not (BASE / 'failure-closed.json').exists()
    prepared = read(BASE / 'focused-prepared.json')
    assert prepared['passed']
    verify(prepared['files'])
    state = read(BASE / 'preparation.json')
    assert state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == ['bridge-restore', 'bridge-build', 'instructions', 'backend-restore', 'backend-build', 'focused']
    terminal(state['supervisor'])
    identities = [state['supervisor']]
    samples = 0
    for run in state['runs']:
        assert run['complete'] and run['code'] == (1 if run['name'] == 'focused' else 0)
        assert run['preflight']['available'] >= (10 if run['name'] == 'focused' else 8) * 1024**3
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            identities.append(identity)
        rows = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(r['rss'] for r in rows) == run['peak_rss']
        for row in rows:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3
            assert row['output_bytes'] <= 1024**3 and row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        samples += len(rows)
    tree = ET.parse(BASE / 'test-results/focused.trx')
    counters = tree.find('.//{*}Counters').attrib
    assert (counters['passed'], counters['failed']) == ('160', '4')
    failures = [dict(name=r.attrib['testName'], message=r.find('.//{*}Message').text) for r in tree.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Failed']
    assert len(failures) == 4 and all('GraphViewsHeldOutputsAndFailureRecover' in f['name'] for f in failures)
    files = dict(prepared['files'])
    for folder in [BASE, TOOLS]:
        for path in folder.rglob('*'):
            if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(folder).parts):
                files[rel(path)] = pin(path)
    save(BASE / 'failure-closed.json', dict(passed=False, files=files, identities=identities, resource_samples=samples,
        counters=counters, failures=failures, inference_started=False,
        reason='Synthetic graph omits IsFused marker and descriptors needed to keep output declarations stable across Reset; all direct destination tests and method-isolation checks pass.'))
    print(json.dumps(dict(closed=pin(BASE / 'failure-closed.json'), samples=samples, identities=len(identities))))


if __name__ == '__main__':
    main()
