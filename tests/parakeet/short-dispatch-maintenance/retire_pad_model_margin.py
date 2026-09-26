"""Use the same retained-output checks for three older, closed runtime probes."""
import sys
import retire_pad_memory_outputs as retire

retire.BASE=retire.ROOT/'artifacts/parakeet-pad-model-margin-retention-20260926'
retire.TARGETS=[
 ('parakeet-isolated-runtime-diagnostic-amd-20260923','parakeet-isolated-runtime-diagnostic-20260923','ecfea393d0475617f32de3b87419cd6354f6d3b61edd4883adbdba877caa0f2c'),
 ('parakeet-dispatch-events-amd-20260923','parakeet-dispatch-events-20260923','c6e1e4d42d3f377f77382a4091ad1150c1a62335a72c7d63a38c728d7a753e19'),
 ('parakeet-wide-runtime-diagnostic-amd-20260923','parakeet-wide-runtime-diagnostic-20260923','fee8b6d9a20917ca24a03300f35720d644b7841ba2ad5408f71392f11bf7642c')]

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('inventory','retire')
    getattr(retire,sys.argv[1])()
