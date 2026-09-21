"""Reuse the established resource auditor with this candidate's exact preparation stages."""
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'convolution-pool/phase_audit.py'
source=path.read_text(encoding='utf8')
old="['bridge-restore', 'bridge-build', 'instructions', 'backend-restore', 'backend-build', 'focused', 'hardware-disabled']"
assert source.count(old)==1
source=source.replace(old,"['core-restore', 'core-build', 'bridge-restore', 'bridge-build', 'instructions', 'backend-restore', 'backend-build', 'focused', 'hardware-disabled']")
old="int(counters['passed']) >= 31";assert source.count(old)==1
source=source.replace(old,"int(counters['passed']) == (214 if name == 'focused' else 50)")
exec(compile(source,str(path),'exec'))
