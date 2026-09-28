"""Consolidate retained September 27 evidence and the closed sigmoid model checks."""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).with_name('dedupe-completed-parakeet-evidence.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'd159a5a27db0c72412eb119ac7b6e50e6692419d29914fd70f7ca6b11128763c'
source = SOURCE.read_text()
assert source.count('1_048_576') == 2
worker = {'__file__': __file__, '__name__': 'closed_september27_evidence'}
exec(compile(source.replace('1_048_576', '4_096'), str(SOURCE), 'exec'), worker)
worker['OUT'] = worker['ARTIFACTS']/'repository-retention-20260923/closed-september27-dedup-20260928'
worker['PATTERNS'] = [
    'parakeet-*-20260927',
    'parakeet-sigmoid-avx512-models-amd-20260928',
]
worker['SUFFIXES'].update({'.cs', '.csproj'})

if __name__ == '__main__': worker['main']()
