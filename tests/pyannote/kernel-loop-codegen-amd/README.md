# Inspect M23 generated code

After complete numerical closure, reuse the unchanged qualified
`SpatialWeightCodegen` driver. Its hash parameters identify the current M22
Core and isolated M23 Core. Both roles use the numerically qualified LayerGraphs
caller: the driver invokes Fixtures and Candidate by reflection and never calls
ModelProbe.Main (whose embedded hash is specific to M23). No helper arithmetic,
product or driver is rebuilt.

Run `run.py prepare`, `stage`, `launch`, `observe`, `collect`, then `audit.py`
using Python 3.13 `-X utf8 -B` from the repository root. Fresh destinations are
`artifacts/pyannote-kernel-loop-codegen-amd-20260922` and
`/dev/shm/lokad-pyannote-kernel-loop-codegen-20260922`.

Two fresh AMD workers run four passes of all 108 captured graph calls. All 432
outputs per role must match the original selected hashes. Only the diagnostic
`DOTNET_JitDisasm=Kernel512*` setting is added; ISA and tiering remain ordinary.
Preserve all emitted tiers, resources and original numerical/ownership checks.
Inspect full-tile branches, broadcasts, FMAs, vector/scalar stack references
and code growth before the fixed component screen. These runs have no timing
score and do not establish a speedup.
