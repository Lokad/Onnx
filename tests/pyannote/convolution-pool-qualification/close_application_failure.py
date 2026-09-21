"""Retain the full-suite CLI timeout before changing its copied test harness."""
import xml.etree.ElementTree as ET
from common import *

BASE = ROOT / 'artifacts/pyannote-convolution-pool-applications-20260921'


def main():
    assert not (BASE / 'failure-closed.json').exists()
    prepared = read(BASE / 'suites-prepared.json')
    verify(prepared['files'])
    state = read(BASE / 'qualification.json')
    assert state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == ['cli-restore', 'cli-build', 'backend-restore', 'backend-build',
        'tensors-restore', 'tensors-build', 'request-focused', 'backend-full']
    terminal(state['supervisor'])
    identities, samples = [state['supervisor']], 0
    for run in state['runs']:
        assert run['complete'] and run['code'] == (1 if run['name'] == 'backend-full' else 0)
        assert run['preflight']['available'] >= (10 if run['name'] in ['request-focused', 'backend-full'] else 8) * 1024**3
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            identities.append(identity)
        rows = [json.loads(s) for s in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(r['rss'] for r in rows) == run['peak_rss']
        for row in rows:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3
            assert row['output_bytes'] <= 1024**3 and row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        samples += len(rows)
    tree = ET.parse(BASE / 'test-results/backend-full.trx')
    counters = tree.find('.//{*}Counters').attrib
    assert (counters['total'], counters['passed'], counters['failed']) == ('3265', '3171', '1')
    failures = [dict(name=r.attrib['testName'], message=r.find('.//{*}Message').text)
        for r in tree.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Failed']
    assert len(failures) == 1 and failures[0]['name'].endswith('CliExitCodeTests.ValidMnistRun_ExitsSuccess')
    assert failures[0]['message'].startswith('CLI timed out: run ')
    assert not (BASE / 'dialogue-output').exists() and not (BASE / 'meetings-run-output').exists()
    files = dict(prepared['files'])
    for folder in [BASE, TOOLS]:
        for path in folder.rglob('*'):
            if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(folder).parts):
                files[rel(path)] = pin(path)
    save(BASE / 'failure-closed.json', dict(passed=False, files=files, identities=identities,
        resource_samples=samples, counters=counters, failures=failures, public_inference_started=False,
        reason='Copied historical CLI test redirects stdout/stderr without draining them; preserve timeout and port the existing test-only correction.'))
    print(json.dumps(dict(closed=pin(BASE / 'failure-closed.json'), samples=samples, identities=len(identities))))


if __name__ == '__main__':
    main()
