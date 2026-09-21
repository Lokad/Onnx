"""Close the stopped campaign without manufacturing a complete timing verdict."""
import hashlib
import importlib.util
import json
import math
from common import *
import numpy as np


def main():
    destination = BASE / 'failure-closed.json'
    assert not destination.exists()
    assert not any((BASE / name).exists() for name in ('analysis.json', 'closed.json', 'completion.json'))
    prepared = read(BASE / 'prepared.json')
    assert prepared['passed']
    verify(prepared['files'])
    for name, wanted in prepared['external_files'].items():
        assert pin(Path(name)) == wanted, name
    finish = read(BASE / 'finish-state.json')
    state = read(BASE / 'processes.json')
    assert finish['complete'] and finish['code'] == 1
    assert state['complete'] and state['code'] == 1
    assert len(finish['stages']) == 1
    assert finish['stages'][0]['phase'] == 'run' and finish['stages'][0]['code'] == 1
    assert finish['stages'][0]['complete']
    assert finish['stages'][0]['worker'] == state['supervisor']
    assert finish['prepared'] == pin(BASE / 'prepared.json')
    for name, wanted in finish['tools'].items():
        assert pin(TOOLS / name) == wanted
    assert [r['name'] for r in state['runs']] == ['0-production', '1-512']
    assert prepared['jobs'] == ['production', '512', '2032', 'ort', 'ort', '2032', '512', 'production']
    assert not any((BASE / f'{i}-{role}').exists() for i, role in enumerate(prepared['jobs']) if i >= 2)
    identities = [finish['supervisor'], state['supervisor']]
    for run in state['runs']:
        assert run['complete']
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
    for identity in identities:
        terminal(identity)

    spec = importlib.util.spec_from_file_location('original_audio_auditor', ROOT / 'tests/audio/comparison/audit.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    manifest = read(INPUT)
    assert (manifest['warmup_passes'], manifest['measured_passes'], len(manifest['cases'])) == (1, 3, 20)
    for case in manifest['cases']:
        pcm = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()

    workers = []
    for index, run in enumerate(state['runs']):
        folder = BASE / run['name'] / 'output'
        paths = sorted(folder.glob('[0-9][0-9][0-9].json'))
        expected_count = (80, 56)[index]
        assert [p.name for p in paths] == [f'{i:03}.json' for i in range(expected_count)]
        rows = [read(p) for p in paths]
        first = {}
        previous_end = 0
        for ordinal, row in enumerate(rows):
            case = manifest['cases'][ordinal % 20]
            assert row['name'] == case['name'] and row['pass'] == ordinal // 20
            assert row['phase'] == ('warmup' if ordinal < 20 else 'measured')
            assert row['ownership'] is True and row['input_sha256'] == case['raw_sha256']
            assert row['frequency'] > 0 and row['end_ticks'] > row['start_ticks'] >= previous_end
            previous_end = row['end_ticks']
            assert math.isfinite(row['seconds']) and row['seconds'] > 0
            assert math.isclose(row['seconds'], (row['end_ticks'] - row['start_ticks']) / row['frequency'], rel_tol=1e-14)
            error = auditor.check_result(row['result'], case['expected'])
            assert math.isclose(row['maximum_centroid_error'], error, rel_tol=1e-12, abs_tol=1e-15)
            assert first.setdefault(case['name'], row['result']) == row['result']
        if index == 0:
            assert run['code'] == 0 and run['application_passed']
            result = read(folder / 'result.json')
            auditor.validate_worker(result, manifest, 'timing')
            assert result['records'] == rows and result['manifest_sha256'] == pin(INPUT)['sha256']
            assert result['engine'] == 'managed' and result['runtime'] == '.NET 10.0.12'
            assert result['processor_count'] == 1 and result['flags'] == {}
            for key, name in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll'), ('runner_sha256', 'AudioBenchmark.dll')]:
                assert result[key] == prepared['roles']['production'][name]['sha256']
        else:
            assert run['code'] == 15 and not run.get('application_passed', False)
            assert not (folder / 'result.json').exists()
            assert 'AssertionError' in run['error']

        samples = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] and max(s['rss'] for s in samples) == run['peak_rss']
        assert run['preflight']['available'] >= 14 * 1024**3
        failures = []
        for ordinal, sample in enumerate(samples):
            assert sample['seconds'] < 1800 and sample['rss'] < 12 * 1024**3
            assert sample['disk'] >= 20 * 1024**3 and sample['output_bytes'] <= 1024**3
            assert len(sample['members']) <= 1
            assert sample['rss'] == sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert member['affinity'] == [2] and run['members'][str(member['pid'])] == member['birth']
            if sample['available'] < 1024**3:
                failures.append(dict(ordinal=ordinal, sample=sample))
        assert len(failures) == index
        if index:
            assert failures[0]['ordinal'] == len(samples) - 1
            assert failures[0]['sample']['available'] == 1041678336
            assert failures[0]['sample']['rss'] == 9923104768
        workers.append(dict(name=run['name'], code=run['code'], completed_requests=len(rows),
                            warmup_requests=sum(r['phase'] == 'warmup' for r in rows),
                            measured_requests=sum(r['phase'] == 'measured' for r in rows),
                            sampled_resources=len(samples), peak_rss=run['peak_rss'],
                            minimum_available=min(s['available'] for s in samples), guard_failures=failures,
                            retained_request_checks_passed=True,
                            final_held_output_check_available=index == 0))

    files = dict(prepared['files'])
    for p in [*BASE.rglob('*'), *TOOLS.iterdir()]:
        if p.is_file():
            files[p.relative_to(ROOT).as_posix()] = pin(p)
    save(destination, dict(evidence_audit_passed=True, campaign_passed=False,
                          attribution_valid=False, selected_for_amd=None,
                          reason='Available RAM below unchanged 1 GiB guard during first 512 MiB worker',
                          planned_requests=640, completed_requests=136, warmup_requests=40, measured_requests=96,
                          ort_workers_started=0, workers=workers, terminal_identities=identities,
                          files=files, external_files=prepared['external_files'],
                          scope='Retained complete and partial request evidence only; no candidate speed, native timing or completed candidate ownership verdict'))
    print(json.dumps(dict(evidence_audit_passed=True, campaign_passed=False, completed_requests=136,
                          resource_samples=sum(w['sampled_resources'] for w in workers), closure=pin(destination))))


if __name__ == '__main__':
    main()
