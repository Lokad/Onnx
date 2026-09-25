# Packed-input final-row proof

One implementation reads the existing 32-column packed layout directly while
preserving the original reduction order, FMA operand order and scalar tail.
It is compared with the original compiled routines from corrected M76; Core
and Data are not rebuilt. No model or performance benchmark runs here.

The frozen case list contains 47 normal, 47 AVX512-disabled and four actual
hardware-disabled cases. It covers partial panels, nonzero destinations, float
special values, both actual weight shapes, all seven affected row counts and
the unchanged even/divisible-by-three routes. Complete route outputs also match
the existing public MatMul. Guard regions and inputs must remain unchanged;
the packed final-row helper must allocate zero managed bytes.

The current M77 timing attribution remains rejected. This proof establishes
arithmetic feasibility only; complete correctness, actual zero-reconstruction
counters and a fresh application comparison remain necessary for a candidate.

Use `C:/Python313/python.exe -X utf8 -B` locally. After preparation and audit
implementation are reviewed, run `run.py prepare`, `stage`, `launch build`,
`observe build`, `collect build`, then `review.py build`. Only after review
passes, run `launch capture`, `observe capture`, `collect capture`, and
`review.py capture`. Build and execute only on the exclusive VM. Every complete
stage is collected and reviewed once; preserve all failed cases separately
from process completion.
