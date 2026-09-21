"""Retain the incorrect assumption that a background GC ends within one request."""
import analyze_v2 as analysis
from common import *

assert not (BASE / 'counter-refusal.json').exists()
try:
    analysis.inspect('sampled-a')
except AssertionError as error:
    assert str(error) == 'GC crosses request boundary'
    rows = [json.loads(line) for line in (BASE / 'sampled-a/events.jsonl').read_text().splitlines()]
    request = next(r for r in analysis.pair_requests(rows) if r['name'] == 'dialogue-30s' and r['pass_index'] == 2)
    gc = next(r for r in analysis.collections_from(rows) if r['count'] == 64)
    assert gc['type'] == 'BackgroundGC' and gc['start_ms'] < request['end_ms'] < gc['end_ms']
    save(BASE / 'counter-refusal.json', dict(passed=False, observed_command='analyze_v2.py', observed_exit_code=1,
        error=str(error), original_analysis=pin(TOOLS / 'analyze_v2.py'), request=request, collection=gc,
        scope='The original CLI successor refused a real background collection crossing a request. Retain its full start/stop events and reconcile counters at collection start instead.'))
    print('Preserved background-collection boundary refusal:', gc['count'])
else:
    raise AssertionError('Original refusal did not reproduce')
