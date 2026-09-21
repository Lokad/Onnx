"""Prospective six-process whole-application comparison, unchanged gates."""
from pathlib import Path

original = Path(__file__).resolve().parents[1] / 'request-comparison/common.py'
source = original.read_text(encoding='utf8')
for before, after in [
    ('artifacts/pyannote-request-comparison-20260921', 'artifacts/pyannote-deferred-views-comparison-20260922'),
    ('artifacts/pyannote-request-contexts-v3-20260921', 'artifacts/pyannote-deferred-views-applications-20260922'),
    ('artifacts/pyannote-optimized-ort-20260921', 'artifacts/pyannote-portable-applications-20260922')]:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
exec(compile(source, str(original), 'exec'))
