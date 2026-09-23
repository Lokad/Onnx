# Isolated Winograd input reuse

Prepare one private AVX512 reduction change against the integrated M34 source.
Each input broadcast serves two adjacent sixteen-channel weight blocks, using
sixteen accumulators for eight tiles. A remaining single output block uses the
original loop. AVX2, transforms, weights, tile count, scratch, epilogue, range
checks, graph dispatch and all selected product files remain unchanged.

The JIT may change broadcast folding or register use. Source reuse alone is
not performance evidence. Require compiled scope, all original numerical
cases and added output-block tails, generated-code review, then a complete-call
screen comparing current Winograd with this candidate. Preserve the existing
10% aggregate / 5% per-form gates, all18repeatability controls and strict
process separation. Full application and package qualification precede integration.

Run C:/Python313/python.exe -X utf8 -B followed by this directory's prepare.py
from repository root. It verifies both complete root/profile closures and all
420selected source files, then writes a separate source tree and patch under
artifacts/pyannote-winograd-output-blocks-source-20260923. It performs no build,
model execution, VM mutation or performance measurement. Existing destinations
are refused; root product changes are refused. The selected Winograd source
must exactly match the previously qualified M33 component before reuse.
