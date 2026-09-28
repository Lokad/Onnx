"""Consolidate identical bytes in eight completed numerical/application campaigns."""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).with_name('dedupe-completed-parakeet-evidence.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'd159a5a27db0c72412eb119ac7b6e50e6692419d29914fd70f7ca6b11128763c'
source = SOURCE.read_text()
assert source.count('1_048_576') == 2
worker = {'__file__': __file__, '__name__': 'closed_transpose_qualification_consolidation'}
exec(compile(source.replace('1_048_576', '65_536'), str(SOURCE), 'exec'), worker)
worker['OUT'] = worker['ARTIFACTS']/'repository-retention-20260923/transpose-closed-qualification-dedup-20260928'
worker['PATTERNS'] = [
    'parakeet-attention-owned-models-amd-20260928',
    'parakeet-attention-owned-app-amd-20260928',
    'parakeet-attention-owned-shared-amd-20260928',
    'parakeet-attention-owned-pyannote-amd-20260928',
    'parakeet-transpose-axis-models-amd-20260928',
    'parakeet-transpose-axis-app-amd-20260928',
    'parakeet-transpose-axis-shared-amd-20260928',
    'parakeet-transpose-axis-pyannote-amd-20260928',
]

if __name__ == '__main__': worker['main']()
