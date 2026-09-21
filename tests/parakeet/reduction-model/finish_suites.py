"""Retain completed backend failures and finish the independent tensor suite."""
import json
import traceback
import xml.etree.ElementTree as ET
from common import *

FIRST = ROOT / 'artifacts/parakeet-reduction-suites-20260921'
BACKEND = ROOT / 'artifacts/parakeet-reduction-suites-v2-20260921'
FINISH = ROOT / 'artifacts/parakeet-reduction-suite-completion-20260921'


def main():
    prepared = read(BACKEND / 'prepared.json')
    verify(prepared['files'])
    state = read(BACKEND / 'processes.json')
    assert state['complete'] and state['code'] == 1 and len(state['runs']) == 1
    terminal(state['supervisor'])
    identities = [state['supervisor']]
    run = state['runs'][0]
    assert run['name'] == 'backend' and run['complete'] and run['code'] == 1
    for pid, birth in run['members'].items():
        identity = dict(pid=int(pid), birth=birth)
        terminal(identity)
        identities.append(identity)
    samples = [json.loads(s) for s in (BACKEND / 'logs/backend.samples.jsonl').read_text().splitlines()]
    assert len(samples) == run['samples'] == 203 and max(s['rss'] for s in samples) == run['peak_rss'] == 4157046784
    assert run['preflight']['available'] >= 10 * 1024**3
    assert all(s['seconds'] < 900 and s['rss'] < 8 * 1024**3 and s['available'] >= 1024**3 and s['disk'] >= 20 * 1024**3
               and s['output_bytes'] <= 1024**3 and all(p['affinity'] == [2] for p in s['members']) for s in samples)
    xml = ET.parse(BACKEND / 'test-results/backend.trx')
    counters = xml.find('.//{*}Counters').attrib
    assert int(counters['failed']) == 2 and int(counters['passed']) == 3099 and int(counters['total']) == 3194
    failures = [dict(name=r.attrib['testName'], message=r.find('.//{*}Message').text) for r in xml.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Failed']
    assert {f['name'] for f in failures} == {'Lokad.Onnx.Backend.Tests.GraphExecutionDinoV3Tests.DinoV3Outputs_MatchFrozenBitHash',
                                           'Lokad.Onnx.Backend.Tests.MatMulKernelAgreementTests.PackedBumpMatchesPacked2Bitwise'}
    backend_files = dict(prepared['files'])
    backend_files.update({rel(p): pin(p) for p in BACKEND.rglob('*') if p.is_file()})
    assert not (BACKEND / 'failure-closed.json').exists()
    save(BACKEND / 'failure-closed.json', dict(evidence_passed=True, suite_passed=False, counters=counters,
         failures=failures, files=backend_files, terminal_identities=identities))
    FINISH.mkdir()
    (FINISH / 'logs').mkdir()
    monitor.BASE = FINISH
    own = psutil.Process()
    current = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    project = FIRST / 'source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'
    try:
        monitor.worker(current, FINISH / 'processes.json', 'tensors',
            ['dotnet', 'test', project, '-c', 'Release', *monitor.FLAGS, '--no-build', '--no-restore',
             '--logger', 'trx;LogFileName=tensors.trx', '--results-directory', FINISH / 'test-results'],
            FIRST / 'source', [0, 1], 10, 8, 900, True, FINISH / 'test-results')
        current['code'] = 0
    except BaseException:
        current.update(code=1, error=traceback.format_exc())
        raise
    finally:
        current['complete'] = True
        save(FINISH / 'processes.json', current)
    verify(prepared['files'])
    tensor = ET.parse(FINISH / 'test-results/tensors.trx')
    tensor_counts = tensor.find('.//{*}Counters').attrib
    assert int(tensor_counts['total']) >= 342
    tensor_failures = [dict(name=r.attrib['testName'], message=r.find('.//{*}Message').text) for r in tensor.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Failed']
    row = current['runs'][0]
    assert row['code'] == (0 if int(tensor_counts['failed']) == 0 else 1)
    for pid, birth in row['members'].items():
        identity = dict(pid=int(pid), birth=birth)
        terminal(identity)
        identities.append(identity)
    tensor_samples = [json.loads(s) for s in (FINISH / 'logs/tensors.samples.jsonl').read_text().splitlines()]
    assert len(tensor_samples) == row['samples'] > 0 and max(s['rss'] for s in tensor_samples) == row['peak_rss']
    assert all(s['seconds'] < 900 and s['rss'] < 8 * 1024**3 and s['available'] >= 1024**3 and s['disk'] >= 20 * 1024**3
               and all(p['affinity'] == [2] for p in s['members']) for s in tensor_samples)
    files = dict(backend_files)
    for p in [*FINISH.rglob('*'), BACKEND / 'failure-closed.json', Path(__file__), BASE / 'closed.json']:
        if p.is_file():
            files[rel(p)] = pin(p)
    save(FINISH / 'closed.json', dict(evidence_passed=True, suites_passed=False, backend_counters=counters,
        backend_failures=failures, tensor_counters=tensor_counts, tensor_failures=tensor_failures,
        files=files, terminal_worker_identities=identities, supervisor=current['supervisor'],
        resource_samples=len(samples)+len(tensor_samples), peak_rss=max(run['peak_rss'],row['peak_rss']),
        scope='Complete suite accounting: two backend compatibility failures remain; no candidate promotion'))
    print(json.dumps(dict(evidence_passed=True, suites_passed=False, backend_counters=counters, tensor_counters=tensor_counts,
                         tensor_failures=tensor_failures, closed=pin(FINISH / 'closed.json'))))


if __name__ == '__main__':
    main()
