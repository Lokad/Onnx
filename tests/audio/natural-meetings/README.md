# Natural meeting ASR preparation

This lane prepares human word references for the same two independently selected
ten-minute AMI excerpts as the [diarization evaluation](../../pyannote/natural-meetings/README.md).
Recognition results are pending. It changes no model, production policy or
numerical tolerance.

Use the official [AMI manual annotations v1.6.2](https://groups.inf.ed.ac.uk/ami/download/),
CC BY 4.0, archive SHA256
`b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`.
The original ISO-8859-1 word XML files preserve four speaker streams per meeting.
Only lexical words entirely inside each fixed crop enter the transcript;
punctuation and non-word events do not. Keep fillers, truncated words and original
spelling. Crossing-boundary words remain explicitly recorded. Input audio is
unchanged.

Sort words by start, end, speaker letter and original XML index. Apply the existing
five-language ASR normalization from `tests/audio/multilingual/common.py`.
This defines a chronological mixed-speaker WER observation. It is not an official
AMI ASR benchmark: overlapping speakers make a single word order ambiguous.
Retain all raw word times and speaker identities alongside the joined reference.
Aggregate only the two independent meeting excerpts; the thirty-second recovery
is correlated and excluded.

`prepare_labels.py --artifact <new-ASR-artifact> --diarization <existing-artifact>`
reads the downloaded archive and reuses the exact existing WAVs by hash. It
requires the lexical interval union from original XML to match the pinned
words-only RTTM references exactly. It extracts only ten named annotation files
and refuses existing outputs. Preparation performs no inference.

Run `python -B -m unittest discover -s tests/audio/natural-meetings` for the
encoding, metadata, ordering, boundary and interval-union tests.

The retained preparation has 1,113 normalized words for ES2004a and 1,348 for
IS1009a. Each excludes one lexical record crossing 600 seconds; thirteen and
twenty annotated truncated lexical records remain included, respectively.
The thirty-second recovery has 45 normalized words and excludes two crossing
records. `audit_labels.py --artifact <directory> --output <new-json>` independently
reconstructs every lexical record with a separate XML DOM parser and verifies
original archive bytes, audio hashes, complete ordering and normalized text.
The current preparation and three policy tests pass. These counts are reference
metadata, not recognition scores.

Attribution: Carletta et al., *The AMI meeting corpus: A pre-announcement* (2006).
Selection and diarization references use the pinned
[pyannote/BUT Speech@FIT setup](https://github.com/pyannote/AMI-diarization-setup/tree/67c2d539286e89f68952d5dcf83912bd9f01dfae).
