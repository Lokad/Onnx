"""Consolidate identical files in eight explicit, completed graph/package campaigns."""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).with_name('dedupe-completed-parakeet-evidence.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'd159a5a27db0c72412eb119ac7b6e50e6692419d29914fd70f7ca6b11128763c'
source = SOURCE.read_text()
assert source.count('1_048_576') == 2
worker = {'__file__': __file__, '__name__': 'closed_transpose_graph_root_consolidation'}
exec(compile(source.replace('1_048_576', '65_536'), str(SOURCE), 'exec'), worker)
worker['OUT'] = worker['ARTIFACTS']/'repository-retention-20260923/transpose-closed-graphs-root-dedup-20260928'
worker['PATTERNS'] = [
    'parakeet-attention-owned-graphs-amd-20260928',
    'parakeet-transpose-axis-graphs-amd-20260928',
    'parakeet-attention-owned-root-recovery-amd-20260928',
    'parakeet-pointwise-tail-root-amd-20260927',
    'parakeet-decoder-lstm-layout-root-amd-20260927',
    'parakeet-decoder-packed-row-root-amd-20260927',
    'parakeet-pointwise-tail-graphs-amd-20260927',
    'parakeet-decoder-lstm-layout-graphs-amd-20260927',
]

if __name__ == '__main__': worker['main']()
