"""Preserve the first synthetic-fixture failure without changing candidate bytes."""
import json
import xml.etree.ElementTree as ET
from common import *


def main():
    target = BASE / 'failure-closed.json'; assert not target.exists()
    prepared = read(BASE / 'prepared.json'); verify(prepared['files'])
    suites = read(BASE / 'suites-prepared.json'); verify(suites['files'])
    state = read(BASE / 'qualification.json'); assert state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == ['cli-restore', 'cli-build', 'backend-restore', 'backend-build',
        'tensors-restore', 'tensors-build', 'request-focused']
    xml = ET.parse(BASE / 'test-results/request-focused.trx'); counters = xml.find('.//{*}Counters').attrib
    assert (counters['total'], counters['passed'], counters['failed']) == ('2', '1', '1')
    failures = [dict(name=r.attrib['testName'], message=r.find('.//{*}Message').text) for r in xml.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Failed']
    assert len(failures) == 1 and failures[0]['name'].endswith('ChangingWindowsAndInterleavedRequestsPreserveOwnedResults')
    assert 'Assert.NotEqual() Failure: Values are equal' in failures[0]['message']
    identities = []; samples = 0
    for filename in ['builds.json', 'qualification.json']:
        controller = read(BASE / filename); terminal(controller['supervisor']); identities.append(controller['supervisor'])
        for run in controller['runs']:
            assert run['complete'] and run['code'] == (1 if run['name'] == 'request-focused' else 0)
            for pid, birth in run['members'].items():
                item = dict(pid=int(pid), birth=birth); terminal(item); identities.append(item)
            rows = [json.loads(line) for line in (BASE / 'logs' / (run['name']+'.samples.jsonl')).read_text().splitlines()]
            assert len(rows) == run['samples'] > 0 and max(r['rss'] for r in rows) == run['peak_rss']
            assert all(r['seconds'] < 900 and r['rss'] < 8*1024**3 and r['available'] >= 1024**3 and r['disk'] >= 20*1024**3
                and r['output_bytes'] <= 1024**3 and all(p['affinity'] == [2] for p in r['members']) for r in rows)
            samples += len(rows)
    assert not (BASE / 'applications-prepared.json').exists()
    files = dict(suites['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(BASE).parts): files[rel(path)] = pin(path)
    files[rel(Path(__file__))] = pin(Path(__file__))
    save(target, dict(passed=False, preparation_passed=True, qualification_passed=False, counters=counters, failures=failures,
        files=files, identities=identities, resource_samples=samples, real_public_workers_started=0,
        diagnosis='The synthetic encoder averages globally centered features, so the intended changing-input stimulus collapses. Preserve failure; use a nonlinear fixture successor with unchanged product DLLs.'))
    print(json.dumps(dict(failure_retained=True, counters=counters, identities=len(identities), resources=samples, closed=pin(target))))


if __name__ == '__main__': main()
