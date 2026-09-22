"""Reuse only independently closed successful stages; never rerun their workers."""
from pathlib import Path
import shutil
from candidate_protocol import pin, read, verified_files, test_results, REQUIRED_TESTS

FAILURE_SHA = 'afdda8b80915238c975c47f18e76e1ce7d99a032beeaa76de1ac5e780bdd601a'
PREFIX = ['sdk-version'] + [name + suffix for name in ('backend', 'tensors', 'cli', 'il-bridge') for suffix in ('-restore', '-build')]
PREFIX += ['il-bridge', 'backend-tests', 'tensors-tests', 'production-pyannote', 'production-parakeet']


def retained(base):
    previous = base / 'predecessor'
    assert pin(previous / 'failure-closed.json')['sha256'] == FAILURE_SHA
    closure = read(previous / 'failure-closed.json')
    assert closure['passed'] and not closure['campaign_passed']
    collected = previous / 'collected'
    collection = read(collected / 'collection.json')
    assert collection['terminal'] and collection['input_error'] is None and collection['code'] == 1
    verified_files(collected, collection['files'])
    for p in collected.rglob('*'):
        if p.is_file():
            assert pin(p) == closure['files']['collected/' + p.relative_to(collected).as_posix()]
    state = read(collected / 'campaign/identity.json')
    assert state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == PREFIX + ['portable-pyannote']
    assert all(r['complete'] and r['code'] == 0 for r in state['runs'][:-1])
    assert state['runs'][-1]['complete'] and state['runs'][-1]['code'] == -6
    assert 'InvalidDataException: Qualified data' in (collected / 'campaign/portable-pyannote/stderr.txt').read_text()
    return collected, collection, state


def verify_prefix(base, campaign):
    collected, collection, state = retained(base)
    original = collected / 'campaign'
    for p in original.rglob('*'):
        name = p.relative_to(original)
        if not p.is_file() or name.as_posix() == 'identity.json' or name.parts[0] == 'portable-pyannote':
            continue
        assert pin(campaign / name) == pin(p), name
    built = read(original / 'built-files.json')
    verified_files(base, built)
    suites = read(original / 'operator-gate.json')['suites']
    for name in ['backend', 'tensors']:
        assert test_results(original / 'test-results' / (name + '.trx'), 3295 if name == 'backend' else 342,
            REQUIRED_TESTS if name == 'backend' else ()) == suites[name]
    assert read(original / 'il-bridge.json')['passed']
    return dict(runs=state['runs'][:-1], seconds=state['seconds'],
        receipt=dict(runs=14, closure=pin(base / 'predecessor/failure-closed.json'),
            collection=pin(collected / 'collection.json'), failed_stage_reused=False, prior_timing_calls=0))


def restore_prefix(base, campaign, absent):
    collected, collection, state = retained(base)
    assert all(absent(identity) for identity in collection['identities'])
    for p in (collected / 'campaign').rglob('*'):
        name = p.relative_to(collected / 'campaign')
        if not p.is_file() or name.as_posix() == 'identity.json' or name.parts[0] == 'portable-pyannote':
            continue
        target = campaign / name
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    return verify_prefix(base, campaign)
