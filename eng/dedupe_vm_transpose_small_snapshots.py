"""Reuse verified snapshot consolidation for remaining files of at least 4 KiB."""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('dedupe_vm_transpose_snapshot_headroom.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '36e727c3d3a0b608617d0486a250775961018ed19e8aecfa237659fd592539d7'
code = source.read_text()
for before, after in [
    ('vm-transpose-snapshot-headroom-20260928', 'vm-transpose-small-snapshots-20260928'),
    ('transpose-snapshot-headroom-20260928.jsonl', 'transpose-small-snapshots-20260928.jsonl'),
    ("return value.replace(before, 'transpose-small-snapshots-20260928.jsonl')",
     "assert value.count('>=65536') == 1\n    return value.replace(before, 'transpose-small-snapshots-20260928.jsonl').replace('>=65536', '>=4096')"),
]:
    assert code.count(before) == 1, before
    code = code.replace(before, after)
exec(compile(code, str(source), 'exec'), globals())
