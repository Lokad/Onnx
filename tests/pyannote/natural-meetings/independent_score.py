"""Small independent DER check using interval sweeps and exact label assignment.

The official pyannote metric remains the reported scorer. This implementation
checks its components for these four-speaker references without using pyannote
or SciPy. Overlap counts each active reference speaker; the collar is zero.
"""
from collections import defaultdict
import math


def score(reference, hypothesis, duration):
    assert math.isfinite(duration) and duration > 0
    ref_labels = sorted({str(s) for _, _, s in reference})
    hyp_labels = sorted({str(s) for _, _, s in hypothesis})
    assert len(ref_labels) <= 12, 'The subset assignment is intended for small meeting references'
    ri = {label: i for i, label in enumerate(ref_labels)}
    hi = {label: i for i, label in enumerate(hyp_labels)}
    events = defaultdict(list)
    events[0.]; events[duration]
    for side, rows in enumerate((reference, hypothesis)):
        for start, end, label in rows:
            assert math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration
            events[start].append((side, str(label), 1))
            events[end].append((side, str(label), -1))
    active = [defaultdict(int), defaultdict(int)]
    intersections = [[0.] * len(hyp_labels) for _ in ref_labels]
    total = missed = false_alarm = minimum = 0.
    points = sorted(events)
    for start, end in zip(points, points[1:]):
        for side, label, delta in events[start]:
            active[side][label] += delta
        assert all(n in (0, 1) for side in active for n in side.values()), 'Overlapping intervals for one label'
        refs = [ri[label] for label, count in active[0].items() if count]
        hyps = [hi[label] for label, count in active[1].items() if count]
        span = end - start
        total += span * len(refs)
        missed += span * max(0, len(refs) - len(hyps))
        false_alarm += span * max(0, len(hyps) - len(refs))
        minimum += span * min(len(refs), len(hyps))
        for r in refs:
            for h in hyps:
                intersections[r][h] += span
    # Each hypothesis label is assigned to at most one reference label. A mask
    # records assigned reference labels, so unused/extra speakers need no dummy
    # costs and the largest gain over all masks is the exact assignment optimum.
    best = {0: (0., ())}
    for h in range(len(hyp_labels)):
        next_best = dict(best)
        for mask, (gain, pairs) in best.items():
            for r in range(len(ref_labels)):
                if mask & (1 << r):
                    continue
                key = mask | (1 << r)
                candidate = gain + intersections[r][h]
                if key not in next_best or candidate > next_best[key][0]:
                    next_best[key] = (candidate, pairs + ((r, h),))
        best = next_best
    correct, pairs = max(best.values(), key=lambda v: v[0])
    confused = minimum - correct
    assert confused >= -1e-9
    confused = max(0., confused)
    der = (missed + false_alarm + confused) / total if total else (1. if false_alarm else 0.)
    return dict(reference_speaker_seconds=total, correct_speaker_seconds=correct,
                missed_speaker_seconds=missed, false_alarm_speaker_seconds=false_alarm,
                confused_speaker_seconds=confused, diarization_error_rate=der,
                mapping=[[ref_labels[r], hyp_labels[h]] for r, h in pairs])
