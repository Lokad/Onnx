"""Reconcile actual call clocks with complete same-process EventPipe records."""
import base64
from collections import Counter
import math

CUSTOM = 'Lokad-Parakeet-MatMul-Diagnostic'
CLR = 'Microsoft-Windows-DotNETRuntime'


def reconcile(value, events, summary):
    assert summary['complete'] and summary['protocol'] == 'all-event-records-v1'
    assert summary['lost'] == 0 and summary['runtime'] == '10.0.8'
    assert len(events) == summary['recorded'] == sum(summary['allCounts'].values())
    assert dict(Counter(e['provider']+':'+e['name'] for e in events)) == summary['allCounts']
    assert sum(e['provider'] == CLR for e in events) == summary['clr_events'] > 0
    assert any(e['provider'] == 'Microsoft-DotNETCore-SampleProfiler' for e in events)
    for index, event in enumerate(events):
        assert event['index'] == index and event['pointerSize'] == 8 and event['pid'] == value['pid']
        assert len(base64.b64decode(event['rawBase64'], validate=True)) == event['rawLength']
        assert math.isfinite(event['ms']) and event['ms'] >= 0
    markers = [e for e in events if e['provider'] == CUSTOM]
    assert len(markers) == 7600 and len(value['diagnostics']) == 3800
    assert all(e['thread'] == value['native_thread'] for e in markers)
    assert all(a['ms'] <= b['ms'] for a, b in zip(markers, markers[1:]))
    scale = 1000/value['frequency']; lower = -math.inf; upper = math.inf
    tolerance = .001  # One microsecond for timestamp conversion/quantization.
    for index, row in enumerate(value['diagnostics']):
        pair = markers[2*index:2*index+2]
        for event, identifier, counter, lo, hi in [
            (pair[0], 1, row['marker'], row['marker'], row['start']),
            (pair[1], 2, row['stop'], row['stop'], row['afterMarker'])]:
            assert event['id'] == identifier
            assert {k: int(v) for k, v in event['payload'].items()} == dict(fixture=0, iteration=index, counter=counter)
            lower = max(lower, event['ms']-hi*scale-tolerance)
            upper = min(upper, event['ms']-lo*scale+tolerance)
    assert lower <= upper and upper-lower <= .01, ('Clock correspondence', lower, upper)
    offset = (lower+upper)/2
    intervals = [dict(ordinal=r['ordinal'], phase=r['phase'], repeat=r['repeat'], call=r['call'],
        begin_ms=r['start']*scale+offset, end_ms=r['stop']*scale+offset,
        elapsed_ms=(r['stop']-r['start'])*scale,
        gc_deltas=[r['after'+str(g)]-r['gc'+str(g)] for g in range(3)],
        pause_delta_ms=(r['pauseAfter']-r['pauseBefore'])/10000)
        for r in value['diagnostics']]
    assert all(0 <= r['begin_ms'] < r['end_ms'] for r in intervals)
    assert all(a['end_ms'] < b['begin_ms'] for a, b in zip(intervals, intervals[1:]))
    return dict(passed=True, diagnostic_only=True, intervals=intervals,
        marker_count=len(markers), records=len(events), lost=0,
        offset_ms=offset, offset_bounds_ms=[lower, upper], clock_tolerance_ms=tolerance,
        counters_scope='GC counters bracket each call and its markers; event intervals determine attribution.')
