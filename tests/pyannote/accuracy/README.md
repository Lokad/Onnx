# Labeled diarization error

This optional Python helper evaluates intervals against human speaker annotations
using the official `pyannote.metrics` 4.1 scorer. It is separate from the native
tensor and timeline conformance lanes. No model-quality acceptance threshold is
implied by a score.

Install `requirements.txt` in an isolated environment and run
`python -m unittest discover -s tests/pyannote/accuracy`. The exact examples check
speaker renaming, missed speech, false alarms, merged speakers, overlap,
silence, boundary error and malformed intervals. No audio or model download is
needed for these checks.

Import `diarization_error.score(reference, hypothesis, duration)`. Both inputs
contain `(start_seconds, end_seconds, anonymous_speaker_label)` tuples. Labels
may be strings or integers. Intervals must be finite and inside the recording;
overlapping intervals for the same speaker are refused. Adjacent intervals are
allowed. Different speakers may overlap.

The scorer optimally matches predicted speakers to reference speakers one to
one. It evaluates the complete recording, includes overlap, and excludes no
time around reference boundaries (zero collar). The returned missed, false
alarm and confused speaker-seconds are summed and divided by reference
speaker-seconds. Overlap contributes once per reference speaker. Following the
upstream convention, a silent reference gives zero without false alarms and
one with any false alarm; component durations remain available. DER may exceed
one when false alarms exceed the amount of reference speech.

Score ordinary and exclusive timelines separately: an exclusive timeline
cannot represent simultaneous speakers. Fixed excerpts from one recording
are correlated boundary diagnostics, not independent accuracy examples.
Report corpus/source identities, evaluation duration, model/configuration and
all error components with any score. Matching native output does not establish
human-label accuracy; favorable accuracy does not erase failed numeric gates.

Metric definition and implementation:
[official pyannote.metrics documentation](https://pyannote.github.io/pyannote-metrics/_modules/pyannote/metrics/diarization.html).
Pinned metric wheel SHA256:
`34a54b7671f61709c1865d0484843e5b46ea3c4e4e5260ab065e5b3156c733d3`.
