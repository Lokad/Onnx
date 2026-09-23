# M29 actual AMD code generation

After expanded numerical and finite-extreme qualification, reuse the exact
SpatialWeightCodegen driver and qualified LayerGraphs caller by reflection.
Two current/candidate512processes each make432ordinary graph calls; all
output hashes, readonly operands, identities and resources must pass.
No rebuild and no performance claim. Only the diagnostic method filter
changes: DOTNET_JitDisasm="Kernel512* ConvBlockedSpatial:Execute". The
[official JIT guide](https://github.com/dotnet/runtime/blob/main/docs/design/coreclr/jit/viewing-jit-dumps.md)
documents space-separated method patterns and class-qualified names.

Retain every raw tier and body. The byte-identical M28 parser keeps exact
known stdout repairs explicit; structural completeness does not pretend
repaired text was uninterleaved. Inspect original Kernel512, new Kernel512Four
and Execute. Capture must include an optimized four-block reduction; inspect
all emitted optimized bodies, arithmetic order, spills, branches, calls,
scalar addressing, stores/tails and code size. Expected full-tile arithmetic
is216FMAs across24accumulators,54broadcasts and36weight loads. Code cleanliness
does not establish a speedup. Original fallback compiled identity is already
proven by the complete normal build inventory.

CPU2workers/CPU0monitor; original12GiBavailable/3GiBtmpfspreflight,8GiBRSS,
1GiBliveavailable/tmpfs/output,900seconds/job,2GiBartifacts. No ISA/tiering
changes. Run Python3.13 -X utf8 -B run.py prepare,stage,launch,observe,collect,
then audit.py and a separate full code review. Fresh artifact
artifacts/pyannote-convolution-four-codegen-amd-20260923; VM
/dev/shm/lokad-pyannote-convolution-four-codegen-20260923. Preserve failures,
collect terminal owners only and never observe closed campaigns.
