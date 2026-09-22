"""Separate confirmed process termination from samples of live workers."""
import subprocess
import time
from candidate_protocol import LIMITS


def validate(event):
    assert event['terminal'] is True and isinstance(event['code'], int)
    assert event['identities'] and 0 <= event['seconds'] < LIMITS['worker_seconds']
    assert event['available'] >= LIMITS['available'] and event['tmpfs_free'] >= LIMITS['tmpfs_free']
    assert event['artifact_bytes'] <= LIMITS['artifact_bytes']


def observe(child, members, identities, absent, seconds, available, tmpfs_free, artifact_bytes):
    if members: return None
    started = time.monotonic()
    try:
        code = child.wait(timeout=.25)
    except subprocess.TimeoutExpired as error:
        raise AssertionError('No sampled members, but the owned root has not terminated') from error
    assert all(absent(dict(pid=int(pid), birth=birth)) for pid, birth in identities.items()), 'An owned descendant is still live'
    event = dict(terminal=True, code=code, identities=dict(identities), seconds=seconds + time.monotonic() - started,
                 available=available, tmpfs_free=tmpfs_free, artifact_bytes=artifact_bytes)
    validate(event)
    return event
