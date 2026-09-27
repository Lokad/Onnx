"""Consolidate verified duplicates in closed attention and predecessor evidence."""
import importlib.util
from pathlib import Path

SOURCE=Path(__file__).with_name('dedupe-completed-parakeet-evidence.py')
spec=importlib.util.spec_from_file_location('closed_evidence_consolidation',SOURCE)
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
assert worker.pin(SOURCE)['sha256']=='d159a5a27db0c72412eb119ac7b6e50e6692419d29914fd70f7ca6b11128763c'
worker.OUT=worker.ARTIFACTS/'repository-retention-20260923/attention-local-dedup-20260928'
worker.PATTERNS=['parakeet-attention*-2026092[78]',
                 'parakeet-pointwise*-20260927',
                 'parakeet-decoder-lstm-layout*-20260927']
# Retain original closure/hash/path/file-identity checks and the 1MiB minimum.
# A live application has no closure and is ineligible. All paths/bytes survive.
# Bind this adapter, including the immutable original-worker digest, in receipt.
worker.__file__=__file__

if __name__=='__main__':worker.main()
