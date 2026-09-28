"""Consolidate verified duplicate files in five terminal sigmoid evidence folders."""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).with_name('dedupe-completed-parakeet-evidence.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'd159a5a27db0c72412eb119ac7b6e50e6692419d29914fd70f7ca6b11128763c'
source = SOURCE.read_text()
assert source.count('1_048_576') == 2
worker = {'__file__': __file__, '__name__': 'closed_sigmoid_evidence_consolidation'}
exec(compile(source.replace('1_048_576', '65_536'), str(SOURCE), 'exec'), worker)
worker['OUT'] = worker['ARTIFACTS']/'repository-retention-20260923/sigmoid-closed-evidence-dedup-20260928'
worker['PATTERNS'] = [
    'parakeet-sigmoid-avx512-build-amd-20260928',
    'parakeet-sigmoid-address-review-20260928',
    'parakeet-sigmoid-residual-diagnostic-amd-20260928',
    'parakeet-transpose-axis-profile-amd-20260928',
    'parakeet-transpose-axis-root-amd-20260928',
]
worker['SUFFIXES'].update({'.cs', '.csproj'})

if __name__ == '__main__': worker['main']()
