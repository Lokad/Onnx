"""Optional labeled diarization scoring, independent of native-output conformance.

Inputs are (start_seconds, end_seconds, anonymous_speaker_label) intervals.
Evaluate the entire recording with zero collar and include overlapping speakers.
The official scorer chooses an optimal one-to-one mapping of speaker labels.
"""
import math
from importlib.metadata import version

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


def annotation(intervals, duration):
    result = Annotation()
    by_speaker = {}
    for index, interval in enumerate(intervals):
        if len(interval) != 3:
            raise ValueError("Expected start, end and speaker label")
        start, end, speaker = interval
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start < end <= duration):
            raise ValueError("Interval must be finite and inside the recording")
        if isinstance(speaker, bool) or not isinstance(speaker, (str, int)):
            raise ValueError("Speaker label must be a string or integer")
        # Tag the type so integer 1 and string '1' remain distinct anonymous labels.
        label = (type(speaker).__name__, speaker)
        by_speaker.setdefault(label, []).append((start, end))
        result[Segment(start, end), index] = label
    for segments in by_speaker.values():
        segments.sort()
        if any(b[0] < a[1] for a, b in zip(segments, segments[1:])):
            raise ValueError("Overlapping intervals for the same speaker")
    return result


def score(reference, hypothesis, duration):
    """Return seconds of each error and DER, without an acceptance threshold.

The denominator counts reference speaker-seconds (overlap counts each speaker).
For a silent reference, the upstream convention returns 0 without false alarms
and 1 with any false alarm; the component durations remain available.
"""
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Recording duration must be finite and positive")
    if version('pyannote.metrics') != '4.1' or version('pyannote.core') != '6.0.1':
        raise ValueError("Use pinned pyannote.metrics 4.1 and pyannote.core 6.0.1")
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    components = metric(annotation(reference, duration), annotation(hypothesis, duration),
                        uem=Timeline([Segment(0, duration)]), detailed=True)
    return {
        'duration_seconds': duration,
        'reference_speaker_seconds': float(components['total']),
        'correct_speaker_seconds': float(components['correct']),
        'missed_speaker_seconds': float(components['missed detection']),
        'false_alarm_speaker_seconds': float(components['false alarm']),
        'confused_speaker_seconds': float(components['confusion']),
        'diarization_error_rate': float(components['diarization error rate']),
        'collar_seconds': 0,
        'overlap_included': True,
    }
