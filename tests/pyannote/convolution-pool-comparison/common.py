"""Use the original normal-runtime comparison contract with the new candidate."""
from pathlib import Path

_original = Path(__file__).resolve().parents[1] / 'request-comparison/common.py'
_source = _original.read_text(encoding='utf8')
for _before, _after in [
    ('artifacts/pyannote-request-comparison-20260921', 'artifacts/pyannote-convolution-pool-comparison-20260921'),
    ('artifacts/pyannote-request-contexts-v3-20260921', 'artifacts/pyannote-convolution-pool-applications-v2-20260921'),
    ('artifacts/pyannote-optimized-ort-20260921', 'artifacts/pyannote-request-contexts-v3-20260921'),
]:
    assert _source.count(_before) == 1
    _source = _source.replace(_before, _after)
exec(compile(_source, str(_original), 'exec'))
