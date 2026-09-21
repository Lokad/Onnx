"""Account exported sampled thread intervals without treating weights as CPU time."""
from collections import defaultdict
import math
import re


def inspect(document, markers):
    assert document['$schema'].endswith('speedscope/file-format-schema.json')
    frames = document['shared']['frames']
    assert all(isinstance(f['name'], str) for f in frames)
    selected_ids = {}
    for name, fragment in markers.items():
        found = [i for i, frame in enumerate(frames) if fragment in frame['name']]
        assert len(found) == 1, (name, found)
        selected_ids[found[0]] = name
    totals, outside, leaves, inclusive, intervals = defaultdict(float), 0., defaultdict(float), defaultdict(float), defaultdict(list)
    profiles, rounding = [], []
    for profile in document['profiles']:
        assert profile['type'] == 'evented' and profile['unit'] == 'milliseconds'
        assert re.fullmatch(r'Thread \(\d+\)', profile['name'])
        stack, opened = [], {}
        previous = profile['startValue']
        assert math.isfinite(previous) and math.isfinite(profile['endValue']) and previous <= profile['endValue']
        total, selected = 0., 0.
        for event in profile['events']:
            value, frame = event['at'], event['frame']
            assert math.isfinite(value) and isinstance(frame, int) and 0 <= frame < len(frames)
            # Float durations exported by TraceEvent may round a close just past
            # the following open. Retain and bound sub-microsecond adjustments.
            if value < previous:
                assert previous-value <= .001, (previous, value)
                rounding.append(previous-value)
                value = previous
            delta = (value-previous)/1000
            marked = [i for i in stack if i in selected_ids]
            assert len(marked) <= 1
            if stack:
                total += delta
                if marked:
                    name = selected_ids[marked[0]]
                    totals[name] += delta
                    selected += delta
                    bucket = frames[stack[-1]]['name']
                    leaf = frames[stack[-2]]['name'] if bucket in ('CPU_TIME', 'UNMANAGED_CODE_TIME') else bucket
                    leaves[(name, leaf, bucket if bucket in ('CPU_TIME', 'UNMANAGED_CODE_TIME') else 'managed')] += delta
                    for text in {frames[i]['name'] for i in stack[stack.index(marked[0]):]}:
                        inclusive[(name, text)] += delta
                else:
                    outside += delta
            if event['type'] == 'O':
                stack.append(frame)
                if frame in selected_ids:
                    assert frame not in opened
                    opened[frame] = value
            else:
                assert event['type'] == 'C' and stack and stack.pop() == frame
                if frame in selected_ids:
                    start = opened.pop(frame)
                    intervals[selected_ids[frame]].append(dict(thread=profile['name'], start_ms=start, end_ms=value))
            previous = value
        assert not stack and not opened and abs(previous-profile['endValue']) <= .001
        assert abs(total-(profile['endValue']-profile['startValue'])/1000) <= .001
        profiles.append(dict(name=profile['name'], seconds=total, selected_seconds=selected, events=len(profile['events'])))
    assert set(totals) == set(markers) and all(totals[n] > 0 for n in markers)
    assert abs(sum(totals.values())+outside-sum(p['seconds'] for p in profiles)) <= 1e-8
    assert abs(sum(leaves.values())-sum(totals.values())) <= 1e-8
    return dict(selected_seconds=dict(totals), outside_marker_seconds=outside, profiles=profiles, intervals=dict(intervals),
        exclusive=[dict(marker=k[0], method=k[1], bucket=k[2], seconds=v) for k, v in sorted(leaves.items(), key=lambda item: -item[1])],
        inclusive=[dict(marker=k[0], method=k[1], seconds=v) for k, v in sorted(inclusive.items(), key=lambda item: -item[1])],
        rounding_adjustments_ms=rounding,
        scope='Estimated sampled thread time under explicit markers; CPU_TIME is an exporter label, not independently measured CPU time.')


def cross_export(speedscope, chromium):
    assert chromium['displayTimeUnit'] == 'ms'
    profiles = {int(re.search(r'\d+', p['name']).group()): p for p in speedscope['profiles']}
    grouped = defaultdict(list)
    for event in chromium['traceEvents']:
        assert event['cat'] == 'sampleEvent' and event['ph'] in ('B', 'E') and event['pid'] == 0
        grouped[event['tid']].append(event)
    assert set(grouped) == set(profiles)
    for tid, profile in profiles.items():
        for a, b in zip(profile['events'], grouped[tid], strict=True):
            assert b['ph'] == ('B' if a['type'] == 'O' else 'E') and b['sf'] == a['frame']
            assert abs(b['ts']/1000-a['at']) <= 1e-8
            frame = chromium['stackFrames'][str(b['sf'])]
            name = speedscope['shared']['frames'][a['frame']]['name']
            assert name == frame['name'] or name == frame['category']+'!'+frame['name']
    return dict(passed=True, profiles=len(profiles), events=sum(len(v) for v in grouped.values()))
