"""Preserve the empty terminal sample and all successful Linux build exits."""
import json
import sys
from candidate_protocol import LIMITS, check_sample, pin, read, write, verified_files
from transport import BASE, PREPARED, ROOT, SITE, checked_local

sys.path.insert(0, str(SITE)); import psutil
prepared, bundle, execution = checked_local()
controller = read(BASE / 'controller/state.json')
assert controller['complete'] and controller['code'] == 1
receipt = read(BASE / 'collected/collection.json')
assert receipt['terminal'] and receipt['input_error'] is None
verified_files(BASE / 'collected', receipt['files'])
state = read(BASE / 'collected/campaign/identity.json')
assert state['complete'] and state['code'] == 1
expected = ['sdk-version'] + [n + '-' + p for n in ['backend', 'tensors', 'cli', 'il-bridge'] for p in ['restore', 'build']]
assert [r['name'] for r in state['runs']] == expected
count = 0; empty = []
for row in state['runs']:
    assert row['complete'] and row['code'] == 0
    samples = [json.loads(s) for s in (BASE / 'collected/campaign' / row['name'] / 'samples.jsonl').read_text().splitlines()]
    assert len(samples) == row['samples'] > 0
    assert max(sum(p['rss'] for p in s['members']) for s in samples) == row['peak_rss']
    for index, sample in enumerate(samples):
        if sample['members']:
            check_sample(sample)
            for p in sample['members']: assert row['members'][str(p['pid'])] == p['birth']
        else:
            assert row['name'] == 'il-bridge-build' and index == len(samples) - 1
            assert sample['seconds'] < row['seconds'] < LIMITS['worker_seconds']
            assert sample['available'] >= LIMITS['available'] and sample['tmpfs_free'] >= LIMITS['tmpfs_free']
            assert sample['artifact_bytes'] <= LIMITS['artifact_bytes']
            empty.append(sample)
    count += len(samples)
assert len(empty) == 1
assert 'Build succeeded.' in (BASE / 'collected/campaign/il-bridge-build/stdout.txt').read_text()
identities = [controller['supervisor']] + [r['child'] for r in controller['stages']]
for identity in identities:
    try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
    except psutil.NoSuchProcess: pass
files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
assert not (BASE / 'failure-closed.json').exists()
write(BASE / 'failure-closed.json', dict(passed=True, campaign_passed=False, failure='Final compiler sample has no live members; compiler exited zero.',
      files=files, successful_build_commands=9, resource_samples=count, rejected_empty_sample=empty[0],
      local_terminal_identities=identities, remote_terminal_identities=receipt['identities'],
      model_checks_started=False, timing_started=False, collection=pin(BASE / 'collected/collection.json')))
print(json.dumps(dict(closed=pin(BASE / 'failure-closed.json'), samples=count, local_identities=len(identities), remote_identities=len(receipt['identities']))))
