"""Close the collected disk-guard failure without scoring an incomplete comparison."""
import datetime
import importlib.util
import json
import math
import shutil
from common import *
from protocol import check_sample, check_result

spec = importlib.util.spec_from_file_location('retained_audio_audit', ROOT/'tests/audio/amd-comparison/audit.py')
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)


def main():
    target = BASE/'failure-closed.json'; assert not target.exists()
    base = BASE/'collected'; receipt = read(base/'collection.json'); state = read(base/'campaign/identity.json')
    assert receipt['terminal'] is True and receipt['code'] == state['code'] == 1 and state['complete'] is True
    assert state['frozen'] == pin(base/'frozen.json') == pin(BASE/'frozen.json')
    for name, wanted in receipt['files'].items():
        assert pin(base/name) == wanted, name
    frozen = read(base/'frozen.json'); manifest = read(base/'manifests/whisper.json')
    assert len(state['runs']) == 5 and [r['name'] for r in state['runs']] == [
        'conformance-00-whisper-managed', 'timing-00-whisper-managed', 'timing-01-whisper-ort',
        'timing-02-whisper-ort', 'timing-03-whisper-managed']
    complete = []
    for run in state['runs'][:-1]:
        folder = base/run['output']; value = read(folder/'worker/result.json')
        validate_records(value, manifest, run['phase']); original.worker_identity(value, manifest, frozen, base, run['engine'])
        samples = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        resources = original.resource_records(run, samples)
        for i, row in enumerate(value['records']):
            assert read(folder/'worker'/f'{i:03}.json') == row
        complete.append(dict(name=run['name'], calls=len(value['records']), resources=resources, result=pin(folder/'worker/result.json')))
    assert [r['calls'] for r in complete] == [20, 80, 80, 80]
    run = state['runs'][-1]; folder = base/run['output']
    assert run['complete'] is True and run['code'] is None and run['error'] == 'AssertionError()'
    assert not (folder/'worker/result.json').exists()
    samples = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    assert len(samples) == run['samples'] == 2064
    for sample in samples[:-1]:
        check_sample(sample)
    last = samples[-1]; assert last['disk'] == 598016 < LIMITS['disk'] and last['available'] == 5086019584 >= LIMITS['available']
    # This checks which single guard failed; the actual failing sample is never changed.
    check_sample(dict(last, disk=LIMITS['disk']))
    assert run['peak_rss'] == max(sum(m['rss'] for m in s['members']) for s in samples)
    assert 0 < run['seconds'] < LIMITS['seconds']
    assert max(b['seconds']-a['seconds'] for a, b in zip(samples, samples[1:])) < 10
    assert all(m['birth'] == run['members'][str(m['pid'])] for s in samples for m in s['members'])
    records = [read(p) for p in sorted((folder/'worker').glob('[0-9]*.json'))]
    assert len(records) == 34
    previous = 0
    for i, row in enumerate(records):
        iteration, index = divmod(i, 20); case = manifest['cases'][index]
        assert row['name'] == case['name'] and row['pass'] == iteration
        assert row['phase'] == ('warmup' if iteration == 0 else 'measured')
        assert row['ownership'] is True and row['input_sha256'] == case['raw_sha256']
        assert all(type(row[k]) is int for k in ['start_ticks', 'end_ticks', 'frequency']) and row['frequency'] > 0
        assert previous <= row['start_ticks'] < row['end_ticks']; previous = row['end_ticks']
        assert math.isclose(row['seconds'], (row['end_ticks']-row['start_ticks'])/row['frequency'], rel_tol=1e-14)
        assert check_result(row['result'], case['expected'], family='whisper') == row['maximum_centroid_error'] == 0
        assert type(row['allocated_bytes']) is int and row['allocated_bytes'] >= 0
        assert len(row['gc_before']) == len(row['gc_after']) == 3
        assert all(type(a) is int and type(b) is int and 0 <= a <= b for a, b in zip(row['gc_before'], row['gc_after']))
    pause = read(BASE/'temporary-service-pause.json'); assert pause['code'] == 0 and pause['after'].count('masked-runtime') == 5
    incident = [json.loads(line) for line in (BASE/'disk-incident-observation.jsonl').read_text().splitlines()]
    assert '2026-09-21T03:40:53+00:00' in incident[0]['stdout'] and 'apt-daily.service' in incident[0]['stdout']
    assert 'Starting packagekit.service' in incident[0]['stdout']
    assert ssh(PRELUDE+'terminal(%r)\nprint("terminal")\n' % receipt['births']).strip() == 'terminal'
    observations = dict(closure_passed=True, campaign_passed=False, reason='disk reserve exhausted during automatic package activity',
        complete_workers=complete, partial_calls=34, partial_records=records, last_sample=last,
        failed_worker_peak_rss=run['peak_rss'], failed_worker_samples=len(samples), limits=LIMITS,
        births=receipt['births'], frozen=pin(base/'frozen.json'), identity=pin(base/'campaign/identity.json'),
        collection=pin(base/'collection.json'), service_pause=pin(BASE/'temporary-service-pause.json'), incident=pin(BASE/'disk-incident-observation.jsonl'))
    write(BASE/'failure-audit.json', observations)
    report = Path(__file__).with_name('disk-failure-20260921.md')
    text = f'''# AMD Whisper comparison stopped by transient disk exhaustion

The fixed comparison is **incomplete**. Conformance completes20/20 calls, followed
by three complete80-call timing workers: managed, ORT and ORT. The final managed
worker stops after34/80 calls when free disk falls to **598,016 bytes**, below the
unchanged33,554,432-byte reserve. Available memory is **5,086,019,584 bytes**;
the other resource guards still pass. This is a disk failure, not a memory failure
or observed application mismatch.

All260 calls from complete workers pass original request, identity, ownership
and resource checks. The34saved partial calls also match the original application
results and their per-request input/ownership checks. They do not establish that
the interrupted request or final held-output check completed. Every prior sample
in the failed worker passes; its2064th sample triggers the stop. All six original
process identities are terminal. No worker is restarted or partial comparison scored.

The final disk samples move from82,522,112 to67,248,128, recover to82,735,104,
then fall to598,016bytes. The journal records apt-daily starting at03:40:53UTC and
PackageKit at03:40:54, when the guard fires. Two roughly61MB package-cache files
were rewritten at03:40:57; free disk subsequently recovered to about83MB. These
observations support attributing the transient exhaustion to package-cache work;
they are not a byte-by-byte allocation trace.

Under the exclusive-VM authorization, apt-daily, apt-daily-upgrade, their timers
and PackageKit are temporarily stopped and masked in `/run`. The saved action
receipt retains prior states and exact restoration commands. Restore those units
after the benchmark window. No model or unique evidence was removed.

The matched AMD timing result remains pending; the Windows Whisper baseline and
completed AMD Parakeet/pyannote baselines retain their original scope. A storage
correction must precede a separately declared replacement comparison. Successful
conformance can be reused after identity verification; this failed timing campaign
must remain visible. e5's independent prepared campaign may proceed after verifying
these terminal identities and closure; its correctness and A/A gates are unchanged.

Frozen SHA256: `{observations['frozen']['sha256']}`. Collection SHA256:
`{observations['collection']['sha256']}`. Artifact:
`artifacts/audio-whisper-amd-20260921`. Every raw record, sample, traceback,
collection receipt, journal observation and service-action receipt is retained.
'''
    with report.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)
    benchmark = ROOT/'BENCHMARK.md'; before = benchmark.read_text(encoding='utf8')
    start = before.index('**Whisper has no AMD timing result:**'); end = before.index('### Audio: Windows Microsoft ONNX Runtime baselines', start)
    section = '''**Whisper's matched AMD comparison is incomplete:** the final managed process
hit the disk-space guard after34/80 calls. The [failure report](tests/audio/whisper-amd/disk-failure-20260921.md)
retains all three completed timing workers, partial results and the coincident
automatic package-cache activity. Memory remained above its guard. No complete
AMD Whisper comparison is claimed; the Windows baseline is below.

'''
    benchmark.write_text(before[:start]+section+before[end:], encoding='utf8', newline='\n')
    support = ROOT/'docs/model-support.md'; text = support.read_text(encoding='utf8')
    text = text.replace('[matched AMD Whisper comparison](../tests/audio/whisper-amd/README.md) is running;',
        '[matched AMD Whisper comparison](../tests/audio/whisper-amd/disk-failure-20260921.md) stopped at a disk-space guard in its final timing process;')
    support.write_text(text, encoding='utf8', newline='\n')
    snapshots = BASE/'failure-snapshots'; snapshots.mkdir()
    for path in [benchmark, support, Path(__file__), ROOT/'tests/audio/amd-comparison/audit.py', ROOT/'tests/audio/amd-comparison/protocol.py']:
        target_path = snapshots/path.relative_to(ROOT); target_path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target_path)
    files = {rel.as_posix(): pin(ROOT/rel) for rel in [p.relative_to(ROOT) for p in sorted(BASE.rglob('*')) if p.is_file()]}
    files[report.relative_to(ROOT).as_posix()] = pin(report)
    write(target, dict(closure_passed=True, campaign_passed=False, reason='disk guard', files=files, births=receipt['births'],
          frozen=observations['frozen'], identity=observations['identity'], collection=observations['collection'],
          utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(closure_passed=True, campaign_passed=False, complete_calls=260, partial_calls=34, closure=pin(target))))


if __name__ == '__main__':
    main()
