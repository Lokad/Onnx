# Inspect M27 generated code

After complete numerical closure, reuse the unchanged qualified
`SpatialWeightCodegen` driver. Its hash parameters identify the current M22
Core and isolated M27 Core. Both roles use the numerically qualified LayerGraphs
caller: the driver invokes Fixtures and Candidate by reflection and never calls
ModelProbe.Main (whose embedded hash is specific to M27). No helper arithmetic,
product or driver is rebuilt.

Run `run.py prepare`, `stage`, `launch`, `observe`, `collect`, then `audit.py`
using Python 3.13 `-X utf8 -B` from the repository root. Fresh destinations are
`artifacts/pyannote-convolution-channel-codegen-amd-20260923` and
`/dev/shm/lokad-pyannote-convolution-channel-codegen-20260923`.

Two fresh AMD workers run four passes of all 108 captured graph calls. All 432
outputs per role must match the original selected hashes. Only the diagnostic
`DOTNET_JitDisasm=Kernel512*` setting is added; ISA and tiering remain ordinary.
Preserve all emitted tiers, resources and original numerical/ownership checks.
Inspect full-tile branches, broadcasts, FMAs, vector/scalar stack references
and code growth before the fixed component screen. These runs have no timing
score and do not establish a speedup.

Expanded numerics c2be39dc and finite-extreme closure a2c99e95 are mandatory.
Inspect first-block zeroing, later-block partial-output loads, channel weight
offsets, missing second-output-block guards, unchanged final non-FMA tails,
vector stack spills and code growth. No timing or speed claim follows.
