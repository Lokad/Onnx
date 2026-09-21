"""Retain the exact runtime-only pairing refusal without changing raw events."""
import analyze
from common import *

assert not (BASE / 'pairing-refusal.json').exists()
path = BASE / 'sampled-a/events.jsonl'
events = [json.loads(line) for line in path.read_text().splitlines()]
try:
    analyze.pair_pauses(events)
except AssertionError as error:
    assert str(error) == 'Overlapping suspension'
    assert events[597]['name'] == events[600]['name'] == 'GC/SuspendEEStart'
    assert events[597]['thread'] != events[600]['thread']
    assert events[599]['name'] == 'GC/RestartEEStart'
    save(BASE / 'pairing-refusal.json', dict(passed=False, observed_command='analyze.py', observed_exit_code=1,
        error=str(error), original_analysis=pin(TOOLS / 'analyze.py'), events=pin(path), context=events[597:602],
        scope='The original CLI analysis refused runtime-only pairing. This read-only replay confirms its exact cause; exporter and captures remain unchanged.'))
    print('Preserved runtime-only overlap refusal; first competing emitters:', events[597]['thread'], events[600]['thread'])
else:
    raise AssertionError('Original refusal did not reproduce')
