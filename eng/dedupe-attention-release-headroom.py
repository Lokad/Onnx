"""Conserve closed predecessor evidence before attention release collection.

The fixed scope excludes campaigns without a complete local file inventory,
all active campaigns, and every VM path. The original worker verifies bytes,
workspace containment and same-volume identity before atomic hard linking.
"""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).with_name('dedupe-completed-parakeet-evidence.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'd159a5a27db0c72412eb119ac7b6e50e6692419d29914fd70f7ca6b11128763c'
worker = {'__file__': __file__, '__name__': 'closed_predecessor_evidence_consolidation'}
exec(compile(SOURCE.read_text(), str(SOURCE), 'exec'), worker)
worker['OUT'] = worker['ARTIFACTS']/'repository-retention-20260923/attention-release-headroom-dedup-20260928'
worker['PATTERNS'] = [
    'parakeet-pad-current-app-amd-20260926',
    'parakeet-pad-current-build-amd-20260926',
    'parakeet-pad-current-graphs-v2-amd-20260926',
    'parakeet-pad-current-models-amd-20260926',
    'parakeet-pad-current-pyannote-amd-20260926',
    'parakeet-pad-current-pyannote-app-amd-20260926',
    'parakeet-pad-current-root-amd-20260926',
    'parakeet-pad-current-screen-amd-20260926',
    'parakeet-pad-current-shared-amd-20260926',
    'parakeet-pad-memory-diagnostic-amd-20260926',
    'parakeet-pad-warmup-diagnostic-amd-20260926',
    'parakeet-padding-anchor-review-20260926',
    'parakeet-rational-sigmoid-app-amd-20260927',
    'parakeet-rational-sigmoid-build-amd-20260927',
    'parakeet-rational-sigmoid-fallback-diagnostic-amd-20260927',
    'parakeet-rational-sigmoid-graphs-amd-20260927',
    'parakeet-rational-sigmoid-models-amd-20260927',
    'parakeet-rational-sigmoid-pyannote-amd-20260927',
    'parakeet-rational-sigmoid-pyannote-app-amd-20260927',
    'parakeet-rational-sigmoid-root-amd-20260927',
    'parakeet-rational-sigmoid-screen-amd-20260927',
    'parakeet-rational-sigmoid-shared-amd-20260927',
    'parakeet-decoder-packed-row-app-amd-20260927',
    'parakeet-decoder-packed-row-contracts-v3-amd-20260927',
    'parakeet-decoder-packed-row-graphs-amd-20260927',
    'parakeet-decoder-packed-row-models-amd-20260927',
    'parakeet-decoder-packed-row-pyannote-amd-20260927',
    'parakeet-decoder-packed-row-pyannote-app-amd-20260927',
    'parakeet-decoder-packed-row-root-amd-20260927',
    'parakeet-decoder-packed-row-screen-v2-amd-20260927',
    'parakeet-decoder-packed-row-shared-amd-20260927',
]

if __name__ == '__main__':
    worker['main']()
