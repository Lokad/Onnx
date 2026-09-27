"""Retain every closed evidence byte while consolidating copies above 256KiB."""
import hashlib
from pathlib import Path

SOURCE=Path(__file__).with_name('dedupe-completed-parakeet-evidence.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='d159a5a27db0c72412eb119ac7b6e50e6692419d29914fd70f7ca6b11128763c'
source=SOURCE.read_text()
# Change only the selection floor and its recorded value. All hash, closure,
# path, identity, atomic-replacement and final byte-verification checks survive.
assert source.count('1_048_576')==2
worker={'__file__':__file__,'__name__':'bounded_closed_qualification_consolidation'}
exec(compile(source.replace('1_048_576','262_144'),str(SOURCE),'exec'),worker)
worker['OUT']=worker['ARTIFACTS']/'repository-retention-20260923/attention-qualification-dedup-20260928'
worker['PATTERNS']=['parakeet-attention*-2026092[78]',
                    'parakeet-pointwise*-20260927',
                    'parakeet-decoder-lstm-layout*-20260927']

if __name__=='__main__':worker['main']()
