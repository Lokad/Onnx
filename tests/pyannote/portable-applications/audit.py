"""Check the complete integrated runtime against predecessor and native outputs."""
import hashlib
import sys
from common import *
import numpy as np


def main():
    assert not (BASE / 'closed.json').exists()
    prepared, state = read(BASE / 'prepared.json'), read(BASE / 'processes.json')
    assert prepared['passed'] and state['complete'] and state['code'] == 0
    verify(prepared['files'])
    assert [run['name'] for run in state['runs']] == ['dialogue', 'meetings-inputs', 'meetings-run']
    identities, resources = [state['supervisor']], []
    for run in state['runs']:
        limit = 900 if run['name'] == 'dialogue' else 3600
        assert run['complete'] and run['code'] == 0 and run['seconds'] < limit
        assert run['preflight']['available'] >= 10 * 1024**3
        samples = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(row['rss'] for row in samples) == run['peak_rss']
        for row in samples:
            assert row['seconds'] < limit and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3
            assert row['disk'] >= 20 * 1024**3 and row['output_bytes'] <= 1024**3 and len(row['members']) <= 1
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        resources.append(dict(name=run['name'], seconds=run['seconds'], samples=len(samples), peak_rss=run['peak_rss']))
    for identity in identities:
        terminal(identity)
    api = module('portable_original_audio_auditor', ROOT / 'tests/audio/comparison/audit.py')
    prior_common = sys.modules['common']
    try:
        sys.modules['common'] = module('portable_original_meeting_common', ROOT / 'tests/pyannote/natural-meetings/common.py')
        meeting_audit = module('portable_original_meeting_auditor', ROOT / 'tests/pyannote/natural-meetings/audit.py')
    finally:
        sys.modules['common'] = prior_common
    manifest = read(INPUT)
    for case in manifest['cases']:
        pcm = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    runtime = BASE / 'runtime'
    assert prepared['core']['sha256'] == CORE and prepared['data']['sha256'] == DATA

    def check_runtime(value, runner, job):
        for key, name in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll'), ('runner_sha256', runner + '.dll')]:
            assert value[key] == pin(runtime / name)['sha256']
        assert value['runtime'] == '.NET 10.0.12' and value['affinity'] == 4 and not value['flags']
        if 'pid' in value:
            assert value['pid'] == next(run for run in state['runs'] if run['name'] == job)['worker']['pid']

    result = read(BASE / 'dialogue-output/result.json')
    api.validate_worker(result, manifest, 'timing')
    check_runtime(result, 'AudioBenchmark', 'dialogue')
    assert result['manifest_sha256'] == pin(INPUT)['sha256']
    old_dialogue = read(PRIOR / 'dialogue-output/result.json')['records']
    dialogue = []
    for i, (row, old) in enumerate(zip(result['records'], old_dialogue, strict=True)):
        assert row == read(BASE / 'dialogue-output' / f'{i:03}.json')
        assert row['name'] == old['name'] and row['pass'] == old['pass'] and row['result'] == old['result']
        assert row['allocated_bytes'] >= 0 and all(b >= a for a, b in zip(row['gc_before'], row['gc_after']))
        dialogue.append({key: row[key] for key in ['name', 'pass', 'phase', 'seconds', 'allocated_bytes', 'maximum_centroid_error']})
    meeting_manifest = read(BASE / 'meetings/manifest.json')
    inputs = read(BASE / 'meetings-inputs-output/inputs.json')
    assert inputs['passed'] and inputs['affinity'] == 4
    assert inputs['cases'] == [{key: case[key] for key in ['name', 'samples', 'pcm_sha256']} for case in meeting_manifest['cases']]
    meetings = read(BASE / 'meetings-run-output/result.json')
    check_runtime(meetings, 'NaturalMeetings', 'meetings-run')
    assert meetings['schema'] == 1 and meetings['engine'] == 'managed' and meetings['held_outputs_unchanged']
    assert meetings['manifest_sha256'] == pin(BASE / 'meetings/manifest.json')['sha256']
    old_meetings = read(PRIOR / 'meetings-run-output/result.json')
    native = read(MEETINGS / 'prior/native.json')
    assert len(meetings['records']) == len(old_meetings['records']) == len(native['records']) == 3
    comparisons = []
    for i, (row, old, ort, case) in enumerate(zip(meetings['records'], old_meetings['records'], native['records'], meeting_manifest['cases'], strict=True)):
        assert row == read(BASE / 'meetings-run-output' / f'{i:02}.json') and row['name'] == old['name'] == ort['name'] == case['name']
        assert row['input_sha256'] == case['pcm_sha256'] and row['ownership'] and row['result'] == old['result']
        assert row['seconds'] == (row['end_ticks'] - row['start_ticks']) / row['frequency'] > 0
        assert row['allocated_bytes'] >= 0 and all(b >= a for a, b in zip(row['gc_before'], row['gc_after']))
        meeting_audit.inspect_public(row['result'], case['samples'])
        comparison = meeting_audit.compare(row['result'], ort['result'])
        assert comparison['passed'] and all(row['result'][key] == ort['result'][key] for key in ['intervals', 'exclusive_intervals'])
        comparisons.append(dict(name=case['name'], **comparison, exact_predecessor=True, exact_native_timelines=True,
            seconds=row['seconds'], allocated_bytes=row['allocated_bytes']))
    analysis = dict(passed=True, core=prepared['core'], data=prepared['data'], public_calls=len(dialogue), meeting_calls=3,
        exact_predecessor_results=True, exact_native_timelines=True, held_outputs_unchanged=True,
        dialogue=dialogue, meetings=comparisons, identities=identities, resources=resources,
        resource_samples=sum(row['samples'] for row in resources), peak_rss=max(row['peak_rss'] for row in resources), scope=prepared['scope'])
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for path in BASE.rglob('*'):
        if path.is_file():
            files[rel(path)] = pin(path)
    save(BASE / 'closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'analysis.json'), identities=identities))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), public_calls=16, meeting_calls=3, resource_samples=analysis['resource_samples'])))


if __name__ == '__main__':
    main()
