"""Close the consumer-identity refusal and independently verify reusable stages."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/pyannote/combined-amd'))
from transport import BASE, PREPARED, SITE, checked_local
from candidate_protocol import pin, read, write, verified_files, check_sample, test_results, REQUIRED_TESTS
sys.path.insert(0, str(SITE))
sys.path.insert(0, str(BASE / 'execution'))
import psutil
from qualify_outputs import pyannote, parakeet


def main():
    assert not (BASE / 'failure-closed.json').exists()
    prepared, bundle, execution = checked_local()
    collected = BASE / 'collected'
    receipt = read(collected / 'collection.json')
    assert receipt['terminal'] and receipt['input_error'] is None and receipt['code'] == 1
    verified_files(collected, receipt['files'])
    assert receipt['payload'] == prepared['payload'] and receipt['execution'] == bundle['execution']
    payload = PREPARED / 'payload'
    verified_files(payload, read(payload / 'payload.json')['files'])
    state = read(collected / 'campaign/identity.json')
    local = read(BASE / 'controller/state.json')
    assert local['complete'] and local['code'] == 1 and state['complete'] and state['code'] == 1
    for identity in [local['supervisor']] + [r['child'] for r in local['stages']]:
        try:
            assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess:
            pass
    assert len(state['runs']) == 15 and all(r['complete'] and r['code'] == 0 for r in state['runs'][:-1])
    failed = state['runs'][-1]
    assert failed['name'] == 'portable-pyannote' and failed['code'] == -6
    assert not (collected / 'campaign/portable-graphs').exists()
    assert 'InvalidDataException: Qualified data' in (collected / 'campaign/portable-pyannote/stderr.txt').read_text()
    samples = 0
    for run in state['runs']:
        rows = [json.loads(s) for s in (collected / 'campaign' / run['name'] / 'samples.jsonl').read_text().splitlines()]
        assert len(rows) == run['samples'] > 0
        assert max(sum(m['rss'] for m in r['members']) for r in rows) == run['peak_rss']
        for row in rows:
            check_sample(row)
            assert all(run['members'][str(m['pid'])] == m['birth'] for m in row['members'])
        samples += len(rows)
    campaign = collected / 'campaign'
    suites = {name: test_results(campaign / 'test-results' / (name + '.trx'), minimum,
        REQUIRED_TESTS if name == 'backend' else ()) for name, minimum in [('backend', 3295), ('tensors', 342)]}
    assert suites == read(campaign / 'operator-gate.json')['suites']
    assert read(campaign / 'il-bridge.json')['passed']
    reports = dict(pyannote=pyannote(payload, campaign / 'production-graphs', 'production'),
        parakeet=parakeet(payload, campaign / 'production-parakeet.json', 'production'))
    assert reports['pyannote']['passed'] and reports['parakeet']['numeric_gate_passed']
    for family in reports:
        assert reports[family] == read(campaign / ('production-' + family + '-audit.json'))
    verified_files(collected, read(campaign / 'built-files.json'))
    analysis = dict(passed=True, campaign_passed=False, cause='GraphQualification hard-coded the previous Data digest.',
        product_failure_observed=False, failed_stage=failed['name'], inference_in_failed_stage=False,
        timing_started=False, reusable_runs=[r['name'] for r in state['runs'][:-1]],
        resource_samples=samples, peak_rss=max(r['peak_rss'] for r in state['runs']), operator_tests=suites,
        previous_rows_reports=reports, collection=pin(collected / 'collection.json'))
    write(BASE / 'failure-analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    write(BASE / 'failure-closed.json', dict(passed=True, campaign_passed=False, files=files,
        analysis=pin(BASE / 'failure-analysis.json'), local_terminal=local['supervisor'], remote_terminal=receipt['identities']))
    print(json.dumps(dict(closed=pin(BASE / 'failure-closed.json'), reusable_runs=14, resource_samples=samples,
        timing_started=False, previous_rows_native_passed=True)))


if __name__ == '__main__':
    main()
