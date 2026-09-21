"""Exact candidate identities for complete public qualification."""
from pathlib import Path

original = Path(__file__).resolve().parents[1] / 'portable-applications/common.py'
source = original.read_text(encoding='utf8')
changes = [
    ('artifacts/pyannote-portable-applications-20260922', 'artifacts/pyannote-deferred-views-applications-20260922'),
    ('artifacts/pyannote-portable-integration-20260922', 'artifacts/pyannote-deferred-views-20260922'),
    ('artifacts/pyannote-sparse-mel-applications-20260921', 'artifacts/pyannote-portable-applications-20260922'),
    ('e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838', '4f22824a7c315334982907f8846dc7e67dd0fddb1aadba4d908c89684285bbd9'),
    ('85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a', '92194232cd60979548cfe480db07a6f7a2e07d9f21dc8c69e133919c31ab4f2d')]
for before, after in changes:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
exec(compile(source, str(original), 'exec'))
