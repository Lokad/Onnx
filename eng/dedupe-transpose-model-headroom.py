"""Consolidate identical retained model evidence before the new model checks."""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).with_name('dedupe-completed-parakeet-evidence.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'd159a5a27db0c72412eb119ac7b6e50e6692419d29914fd70f7ca6b11128763c'
source = SOURCE.read_text()
assert source.count('1_048_576') == 2
worker = {'__file__': __file__, '__name__': 'closed_model_evidence_consolidation'}
exec(compile(source.replace('1_048_576', '65_536'), str(SOURCE), 'exec'), worker)
worker['OUT'] = worker['ARTIFACTS']/'repository-retention-20260923/transpose-model-headroom-dedup-20260928'
worker['PATTERNS'] = [
    'parakeet-attention-owned-models-amd-20260928',
    'parakeet-decoder-lstm-layout-models-amd-20260927',
    'parakeet-decoder-packed-row-models-amd-20260927',
    'parakeet-direct-depthwise-models-amd-20260925',
    'parakeet-first-use-kernels-models-amd-v2-20260923',
    'parakeet-inclusive-packing-models-amd-20260924',
    'parakeet-observed-dense-where-models-amd-20260924',
    'parakeet-owned-batch-isolation-models-amd-20260925',
    'parakeet-owned-packed-weight-models-amd-20260925',
    'parakeet-packed-final-row-models-amd-20260925',
    'parakeet-pad-current-models-amd-20260926',
    'parakeet-pointwise-tail-models-amd-20260927',
    'parakeet-prepared-recurrence-models-amd-20260924',
    'parakeet-rational-sigmoid-models-amd-20260927',
    'parakeet-slice-dense-conversion-models-amd-20260925',
    'parakeet-slice-materialization-models-amd-20260924',
    'parakeet-validated-composition-models-amd-20260924',
    'parakeet-wide-entry-first-use-models-amd-20260923',
    'parakeet-transpose-axis-build-recovery-amd-20260928',
]

if __name__ == '__main__': worker['main']()
