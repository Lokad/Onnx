"""Fixed AMI selection and annotation policy, independent of inference outputs."""
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib
import json

REVISION = '67c2d539286e89f68952d5dcf83912bd9f01dfae'
MEETINGS = ['ES2004a', 'IS1009a']
DURATION = 600
CORE = '187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4'
DATA = '809242b58725c6ae47514cc3908ef59ffafae6be36bb6e2fba20144d9a975af5'


def pin(path):
    with Path(path).open('rb') as stream:
        return dict(bytes=Path(path).stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def select(text):
    names = text.split()
    if len(set(names)) != len(names):
        raise ValueError('Duplicate meeting identifier')
    result = []
    for site in ['ES', 'IS']:
        available = sorted(n for n in names if n.startswith(site) and n.endswith('a'))
        if not available:
            raise ValueError('Missing selected site')
        result.append(available[0])
    if result != MEETINGS:
        raise ValueError('Pinned metadata selection differs')
    return result


def decimal(value):
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError('Invalid annotation time') from error
    if not number.is_finite():
        raise ValueError('Nonfinite annotation time')
    return number


def annotations(rttm, uem, meeting, duration=DURATION):
    fields = uem.split()
    if len(fields) != 4 or fields[:2] != [meeting, '1'] or decimal(fields[2]) != 0 or decimal(fields[3]) < duration:
        raise ValueError('UEM does not cover the full fixed excerpt')
    groups = defaultdict(list)
    for line in rttm.splitlines():
        f = line.split()
        if len(f) != 10 or f[:3] != ['SPEAKER', meeting, '1'] or f[7] == '<NA>':
            raise ValueError('Invalid RTTM identity or layout')
        start, length = decimal(f[3]), decimal(f[4])
        if start < 0 or length <= 0:
            raise ValueError('Invalid RTTM interval')
        end = min(start + length, Decimal(duration))
        if start < end:
            groups[f[7]].append((start, end))
    merged = []
    for speaker, intervals in groups.items():
        current = []
        for start, end in sorted(intervals):
            if current and start <= current[-1][1]:
                current[-1] = (current[-1][0], max(end, current[-1][1]))
            else:
                current.append((start, end))
        merged.extend((float(start), float(end), speaker) for start, end in current)
    merged.sort()
    if not merged:
        raise ValueError('Empty reference')
    return merged


def coverage(intervals, duration=DURATION):
    events = defaultdict(int)
    for start, end, speaker in intervals:
        if not 0 <= start < end <= duration:
            raise ValueError('Interval outside excerpt')
        events[start] += 1
        events[end] -= 1
    active, previous = 0, 0.
    spans = defaultdict(float)
    for moment, delta in sorted(events.items()):
        spans[active] += moment - previous
        active += delta
        previous = moment
        if active < 0:
            raise ValueError('Invalid event count')
    if active:
        raise ValueError('Unclosed interval')
    spans[0] += duration - previous
    return dict(speakers=len({row[2] for row in intervals}), intervals=len(intervals),
                speech_seconds=sum(v for k, v in spans.items() if k > 0),
                overlap_seconds=sum(v for k, v in spans.items() if k > 1),
                reference_speaker_seconds=sum(k * v for k, v in spans.items()),
                seconds_by_active_speakers={str(k): v for k, v in sorted(spans.items())})
