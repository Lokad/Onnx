"""Require both long meetings and recovery to preserve native decisions."""
from protocol import pin, read
from meeting_protocol import inspect_worker, compare


def audit_meetings(base, campaign):
    manifest_path = base / 'meetings/manifest.json'
    manifest = read(manifest_path)
    inputs = read(campaign / 'meetings-inputs/output/inputs.json')
    assert inputs['passed'] and inputs['affinity'] == 4
    assert inputs['cases'] == [dict(name=c['name'], samples=c['samples'], pcm_sha256=c['pcm_sha256']) for c in manifest['cases']]
    output = campaign / 'meetings-run/output/result.json'
    result = read(output)
    inspect_worker(result, manifest, 'managed')
    assert result['manifest_sha256'] == pin(manifest_path)['sha256']
    assert result['runner_sha256'] == pin(base / 'runtimes/candidate/NaturalMeetings.dll')['sha256']
    assert [r['name'] for r in result['records']] == ['ES2004a', 'IS1009a', 'ES2004a-recovery30']
    comparisons = []
    for label in ['native', 'portable-reference']:
        reference = read(base / 'meetings' / (label + '.json'))
        assert len(reference['records']) == len(result['records']) == 3
        for actual, expected in zip(result['records'], reference['records'], strict=True):
            assert actual['name'] == expected['name']
            comparisons.append(dict(name=actual['name'], reference=label, **compare(actual['result'], expected['result'])))
    selected=read(base/'evidence/selected-meetings.json')
    assert [r['result'] for r in result['records']]==[r['result'] for r in selected['records']]
    return dict(complete_selected_results_exact=True,passed=all(r['passed'] for r in comparisons), comparisons=comparisons,
        calls=3, input=pin(campaign / 'meetings-inputs/output/inputs.json'), result=pin(output),
        maximum_centroid_error=max(r['maximum_centroid_error'] for r in comparisons))
