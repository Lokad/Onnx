"""Original owned-process checks with exact Data-only preparation stages and suite counts."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'convolution-pool/phase_audit.py'
source = path.read_text(encoding='utf8')
for old, new in [
    ("['bridge-restore', 'bridge-build', 'instructions', 'backend-restore', 'backend-build', 'focused', 'hardware-disabled']",
     "['data-restore', 'data-build', 'bridge-restore', 'bridge-build', 'instructions', 'backend-restore', 'backend-build', 'focused', 'hardware-disabled']"),
    ("int(counters['passed']) >= 31", "int(counters['passed']) == 89")]:
    assert source.count(old) == 1
    source = source.replace(old, new)
exec(compile(source, str(path), 'exec'))
