"""Original resource audit, with this candidate's exact stage and test counts."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'convolution-pool/phase_audit.py'
source = path.read_text(encoding='utf8')
changes = [
    ("['bridge-restore', 'bridge-build', 'instructions', 'backend-restore', 'backend-build', 'focused', 'hardware-disabled']",
     "['core-restore', 'core-build', 'bridge-restore', 'bridge-build', 'instructions', 'backend-restore', 'backend-build', 'focused', 'hardware-disabled']"),
    ("int(counters['passed']) >= 31", "int(counters['passed']) == (358 if name == 'focused' else 144)")]
for old, new in changes:
    assert source.count(old) == 1
    source = source.replace(old, new)
exec(compile(source, str(path), 'exec'))
