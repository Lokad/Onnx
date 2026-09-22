# Isolated convolution channel-block prototype

Reuse a sixteen-input-channel weight range across an output row when c>=64.
Kernel512 retains its six-position/two-output-block tile, but visits input
channel blocks before spatial tiles. Later blocks load the previous output
accumulators before continuing the original ic/ky/kx FMA chain. The first
block starts at zero. Missing second output blocks never cause an output
read. Smaller channel counts retain a single reduction; Kernel256 is exact.

The [source review](../integration-review/ort-channel-blocks-20260923.md)
explains the hypothesis and its limits. Partial output traffic and JIT layout
may outweigh weight reuse. No correctness or speed claim exists yet.

Preparation copies all416 qualified root files and changes one isolated file:
`artifacts/pyannote-convolution-channel-blocks-20260923/source/src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs`.
All root source hashes remain unchanged. Prepared receipt SHA256:
`77c31d6a4023a024c18da710aa22cb1e094829e8dc12e27948b411b4d11b7edf`.
Patch SHA256:
`badfba91c1d41ed0b7982f6feca98c73d3bd3be219a3bbaaba90b5c088d86d4f`.
Preparation is complete; do not rerun prepare.py into that destination.

Next use a fresh ordinary AMD build/inventory lane. Only Kernel512 may change:
3,162 other Core methods,697 Data methods and public declarations must remain
exact. Then retain every inherited raw/wide/stride-two/captured-layer check and
add c64/80/128/256 raw coverage: the original c16/32 raw families cannot test
the new channel split. Both AVX2 and AVX512 modes, sentinels, held outputs,
nonfinite fallback and final non-FMA spatial tails remain mandatory.

After actual code inspection, keep the fixed108-graph complete-call screen:
all17,184 clocks,32 repeatability controls and12 speed gates. No candidate is
integrated before full product/package/model/native/meeting/application
qualification. Current Pyannote/Parakeet ORT figures remain unchanged.
