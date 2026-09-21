"""Audit exact public decisions, native bounds, allocation observations and resources."""
import hashlib
import importlib.util
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from common import *
import numpy as np


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def main():
    assert not (BASE / 'closed.json').exists()
    prepared = read(BASE / 'applications-prepared.json'); assert prepared['passed']; verify(prepared['files'])
    state = read(BASE / 'qualification.json'); assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == ['cli-restore', 'cli-build', 'backend-restore', 'backend-build', 'tensors-restore',
        'tensors-build', 'request-focused', 'backend-full', 'tensors-full', 'dialogue', 'meetings-inputs', 'meetings-run']
    identities = []; samples = 0; resources = []
    for name in ['builds.json', 'qualification.json']:
        controller = read(BASE / name); assert controller['complete'] and controller['code'] == 0
        terminal(controller['supervisor']); identities.append(controller['supervisor'])
        for run in controller['runs']:
            assert run['complete'] and run['code'] == 0 and run['samples'] > 0
            for pid, birth in run['members'].items():
                identity = dict(pid=int(pid), birth=birth); terminal(identity); identities.append(identity)
            rows = [json.loads(line) for line in (BASE / 'logs' / (run['name']+'.samples.jsonl')).read_text().splitlines()]
            assert len(rows) == run['samples'] and max(r['rss'] for r in rows) == run['peak_rss']
            inference = run['name'] in ['dialogue', 'meetings-inputs', 'meetings-run']
            seconds = 3600 if run['name'].startswith('meetings-') else 900
            minimum = 10 if inference or run['name'] in ['request-focused', 'backend-full', 'tensors-full'] else 8
            assert run['preflight']['available'] >= minimum * 1024**3
            for row in rows:
                assert row['seconds'] < seconds and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3
                assert row['output_bytes'] <= 1024**3 and (not inference or len(row['members']) <= 1)
                assert row['rss'] == sum(p['rss'] for p in row['members'])
                assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
            samples += len(rows)
            resources.append(dict(name=run['name'], samples=len(rows), peak_rss=run['peak_rss'], seconds=run['seconds'], preflight=run['preflight']))
    suites = read(BASE / 'suites.json')
    assert [s['name'] for s in suites] == ['request-focused', 'backend-full', 'tensors-full']
    for suite in suites:
        xml = ET.parse(BASE / 'test-results' / (suite['name']+'.trx'))
        assert xml.find('.//{*}Counters').attrib == suite['counters'] and int(suite['counters']['failed']) == 0
    focused = ET.parse(BASE / 'test-results/request-focused.trx')
    assert {r.attrib['testName'].split('.')[-1] for r in focused.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Passed'} == {
        'ChangingWindowsAndInterleavedRequestsPreserveOwnedResults', 'CancellationInvalidInputAndGraphFailureRecoverWithExistingContexts'}
    api_audit = module('original_audio_auditor', ROOT / 'tests/audio/comparison/audit.py')
    prior_common = sys.modules['common']
    try:
        sys.modules['common'] = module('original_meeting_common', ROOT / 'tests/pyannote/natural-meetings/common.py')
        meeting_audit = module('original_meeting_auditor', ROOT / 'tests/pyannote/natural-meetings/audit.py')
    finally: sys.modules['common'] = prior_common
    manifest = read(INPUT)
    for case in manifest['cases']:
        pcm = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    result = read(BASE / 'dialogue-output/result.json'); api_audit.validate_worker(result, manifest, 'timing')
    assert result['manifest_sha256'] == pin(INPUT)['sha256'] and result['runtime'] == '.NET 10.0.12'
    runtime = BASE / 'application-runtime'
    def runtime_check(value, runner):
        for key, name in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll'), ('runner_sha256', runner+'.dll')]:
            assert value[key] == pin(runtime / name)['sha256']
    runtime_check(result, 'AudioBenchmark')
    previous = read(PUBLIC / 'process/0-candidate/output/result.json')
    prior_results = {r['name']: r['result'] for r in previous['records']}
    for i, row in enumerate(result['records']):
        assert row == read(BASE / 'dialogue-output' / f'{i:03}.json') and row['result'] == prior_results[row['name']]
        assert row['allocated_bytes'] >= 0 and all(b >= a for a, b in zip(row['gc_before'], row['gc_after']))
    dialogue = []
    prior_rows = [r for worker in ['0-candidate', '3-candidate'] for r in read(PUBLIC / 'process' / worker / 'output/result.json')['records']]
    for case in manifest['cases']:
        old = [r for r in prior_rows if r['name'] == case['name'] and r['phase'] == 'measured']
        new = [r for r in result['records'] if r['name'] == case['name'] and r['phase'] == 'measured']
        a = statistics.fmean(r['allocated_bytes'] for r in old); b = statistics.fmean(r['allocated_bytes'] for r in new)
        dialogue.append(dict(name=case['name'], prior_calls=len(old), candidate_calls=len(new), prior_allocated_mean=a, candidate_allocated_mean=b,
            allocation_ratio=b/a, candidate_seconds=[r['seconds'] for r in new]))
    meeting_manifest = read(BASE / 'meetings/manifest.json')
    inputs = read(BASE / 'meetings-inputs-output/inputs.json')
    assert inputs['passed'] and inputs['affinity'] == 4
    assert inputs['cases'] == [{k:c[k] for k in ['name', 'samples', 'pcm_sha256']} for c in meeting_manifest['cases']]
    meetings = read(BASE / 'meetings-run-output/result.json'); runtime_check(meetings, 'NaturalMeetings')
    assert meetings['schema'] == 1 and meetings['engine'] == 'managed' and meetings['held_outputs_unchanged'] and meetings['affinity'] == 4
    assert meetings['runtime'] == '.NET 10.0.12' and not meetings['flags'] and meetings['manifest_sha256'] == pin(BASE / 'meetings/manifest.json')['sha256']
    old_meetings = read(MEETINGS / 'output-run/result.json'); native = read(MEETINGS / 'prior/native.json')
    assert len(meetings['records']) == len(old_meetings['records']) == len(native['records']) == 3
    comparisons = []
    for i, (row, old, ort, case) in enumerate(zip(meetings['records'], old_meetings['records'], native['records'], meeting_manifest['cases'], strict=True)):
        assert row == read(BASE / 'meetings-run-output' / f'{i:02}.json') and row['name'] == old['name'] == ort['name'] == case['name']
        assert row['input_sha256'] == case['pcm_sha256'] and row['ownership'] and row['result'] == old['result']
        assert row['seconds'] == (row['end_ticks'] - row['start_ticks']) / row['frequency'] > 0
        assert row['allocated_bytes'] >= 0 and all(b >= a for a, b in zip(row['gc_before'], row['gc_after']))
        meeting_audit.inspect_public(row['result'], case['samples']); comparison = meeting_audit.compare(row['result'], ort['result']); assert comparison['passed']
        assert all(row['result'][k] == ort['result'][k] for k in ['intervals', 'exclusive_intervals'])
        comparisons.append(dict(name=case['name'], **comparison, exact_predecessor=True, exact_native_timelines=True,
            allocated_bytes=row['allocated_bytes'], prior_allocated_bytes=old['allocated_bytes'], allocation_ratio=row['allocated_bytes']/old['allocated_bytes'], seconds=row['seconds']))
    allocation_passed = dialogue[0]['allocation_ratio'] < 1 and all(c['allocation_ratio'] < 1 for c in comparisons[:2])
    analysis = dict(passed=True, public_calls=16, meeting_calls=3, exact_predecessor_results=True, allocation_reduced=allocation_passed,
        dialogue=dialogue, meetings=comparisons, suites=suites, resources=resources, resource_samples=samples,
        peak_rss=max(r['peak_rss'] for r in resources), identities=identities,
        instruction_changes=[{k:v for k,v in r.items() if k not in ['normalized_methods', 'candidate_methods']} for r in read(BASE / 'instructions.json')['observations']],
        scope='Output/allocation qualification only; historical timings are not a matched performance comparison; no AMD or production promotion.')
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(BASE).parts): files[rel(path)] = pin(path)
    for path in TOOLS.iterdir():
        if path.is_file(): files[rel(path)] = pin(path)
    save(BASE / 'closed.json', dict(passed=True, allocation_reduced=allocation_passed, files=files, identities=identities, analysis=pin(BASE / 'analysis.json')))
    print(json.dumps({k:analysis[k] for k in ['passed', 'public_calls', 'meeting_calls', 'allocation_reduced', 'resource_samples', 'peak_rss']}))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'))))


if __name__ == '__main__': main()
