"""Preserve the 2 GiB suite stop; execute unchanged tests under explicit 8 GiB limits."""
import json
import traceback
import xml.etree.ElementTree as ET
from common import *

OLD = ROOT / 'artifacts/parakeet-reduction-suites-20260921'
NEW = ROOT / 'artifacts/parakeet-reduction-suites-v2-20260921'


def main():
    old = read(OLD / 'processes.json')
    assert old['complete'] and old['code'] == 1
    terminal(old['supervisor'])
    identities = [old['supervisor']]
    for run in old['runs']:
        assert run['complete']
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            identities.append(identity)
    assert [r['name'] for r in old['runs']] == ['cli-restore', 'cli-build', 'backend-restore', 'backend-build', 'tensors-restore', 'tensors-build', 'backend-tests']
    assert all(r['code'] == 0 for r in old['runs'][:-1]) and old['runs'][-1]['code'] == 15
    samples = [json.loads(s) for s in (OLD / 'logs/backend-tests.samples.jsonl').read_text().splitlines()]
    assert len(samples) == old['runs'][-1]['samples'] == 34
    assert all(s['rss'] < 2 * 1024**3 for s in samples[:-1])
    assert samples[-1]['rss'] == 2153668608 > 2 * 1024**3 and samples[-1]['available'] == 13232267264
    assert all(s['seconds'] < 900 and s['available'] >= 1024**3 and s['disk'] >= 20 * 1024**3
               and all(p['affinity'] == [2] for p in s['members']) for s in samples)
    assert not (OLD / 'test-results/backend.trx').exists()
    preparation = read(OLD / 'prepared.json')
    verify(preparation['source_files'])
    runtime = BASE / 'runtime'
    for label in ('cli', 'backend', 'tensors'):
        folder = OLD / 'source' / ('src/Lokad.Onnx.CLI' if label == 'cli' else 'tests/Lokad.Onnx.' + ('Backend' if label == 'backend' else 'Tensors') + '.Tests') / 'bin/Release/net10.0'
        for name, expected in preparation['runtime'].items():
            assert pin(folder / (name + '.dll')) == expected == pin(runtime / (name + '.dll'))
    old_files = {rel(p): pin(p) for p in OLD.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(OLD).parts)}
    assert not (OLD / 'failure-closed.json').exists()
    save(OLD / 'failure-closed.json', dict(evidence_passed=True, suite_passed=False,
        reason='Backend process tree exceeded declared 2 GiB limit before a TRX result; tensors not started',
        failing_sample=samples[-1], terminal_identities=identities, files=old_files))
    NEW.mkdir()
    (NEW / 'logs').mkdir()
    old_files[rel(OLD / 'failure-closed.json')] = pin(OLD / 'failure-closed.json')
    old_files[rel(Path(__file__))] = pin(Path(__file__))
    save(NEW / 'prepared.json', dict(passed=True, files=old_files, limits=dict(preflight_gib=10, rss_gib=8, seconds=900),
        rationale='Earlier 2 GiB build-scale bound stopped testhost; established full-suite 8 GiB envelope, no binary/test/tolerance changes'))
    monitor.BASE = NEW
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    results = []
    try:
        for label in ('backend', 'tensors'):
            name = 'Lokad.Onnx.' + ('Backend' if label == 'backend' else 'Tensors') + '.Tests'
            project = OLD / 'source/tests' / name / (name + '.csproj')
            row = monitor.worker(state, NEW / 'processes.json', label,
                ['dotnet', 'test', project, '-c', 'Release', *monitor.FLAGS, '--no-build', '--no-restore',
                 '--logger', 'trx;LogFileName=' + label + '.trx', '--results-directory', NEW / 'test-results'],
                OLD / 'source', [0], 10, 8, 900, True, NEW / 'test-results')
            counters = ET.parse(NEW / 'test-results' / (label + '.trx')).find('.//{*}Counters').attrib
            assert int(counters['failed']) == 0 and int(counters['passed']) > 300
            results.append(dict(suite=label, counters=counters))
            print(label, counters, flush=True)
        verify(old_files)
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(NEW / 'processes.json', state)
    new_identities = []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            new_identities.append(identity)
        samples = [json.loads(s) for s in (NEW / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(s['rss'] for s in samples) == run['peak_rss']
        assert run['preflight']['available'] >= 10 * 1024**3
        assert all(s['rss'] < 8 * 1024**3 and s['seconds'] < 900 and s['available'] >= 1024**3 and s['disk'] >= 20 * 1024**3
                   and s['output_bytes'] <= 1024**3 and all(m['affinity'] == [2] for m in s['members']) for s in samples)
    files = dict(old_files)
    files.update({rel(p): pin(p) for p in NEW.rglob('*') if p.is_file()})
    save(NEW / 'closed.json', dict(passed=True, files=files, results=results, worker_identities=new_identities,
        supervisor=state['supervisor'], resource_samples=sum(r['samples'] for r in state['runs']),
        peak_rss=max(r['peak_rss'] for r in state['runs']), failed_predecessor=pin(OLD / 'failure-closed.json'),
        model_closure=pin(BASE / 'closed.json')))
    print(json.dumps(dict(passed=True, results=results, closed=pin(NEW / 'closed.json'))))


if __name__ == '__main__':
    main()
