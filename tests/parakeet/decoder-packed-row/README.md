# One prepared-weight reader for the observed decoder projection

This isolated candidate follows the closed [decoder observation](../decoder-projection-observation-results/README.md).
The release remains Core `65f15a41` / Data `da72ca54`. Only two existing Core
methods change: `ResolvePackedKernel` admits a fresh mapping for one row in
the existing output-width-at-least-8,192 territory; `RunPreparedPackedRows`
calls one new internal `PreparedSingleRowKernel.Multiply` for that case.
The reader uses the existing 32-column panels with four AVX2 accumulators,
the same B/A FMA order, ascending reduction, final stores and scalar tail.
No new packing format, wider vector, prefetch or unrolling variant.

The existing mapping guards and shared prepared dispatch remain. Their metadata
and loop bookkeeping can change small public-call allocation totals; record
those totals and include that work in the later public-call comparison. The
raw reader must allocate zero bytes. Output payload ownership stays unchanged.

Preparation checks the qualified root's 439 source inputs and both prerequisite
closures. It copies those inputs into a fresh isolated artifact and adds the
reader. The source patch is reversible and changes exactly two method bodies.
Only Core and one common contract consumer are built. Data and all dependencies
are the qualified bytes; the existing compiled-inventory bridge is reused.
The consumer is compiled against the current Core and runs unchanged with both
products. Its test assembly name uses the existing test-only internal access.

Before contracts run, the compiled inventory must prove all 3,281 other Core
methods and all 697 Data methods unchanged, exactly one internal method added,
and identical public surface, assembly metadata and existing method flags.

Correctness uses six fresh processes: current/candidate in normal mode, then
current/candidate with AVX512 disabled, then current/candidate with hardware
intrinsics disabled. Every process checks 45 public cases: widths around the
dispatch and panel boundaries, one/even/three/other-odd rows, shared-B batches,
scalar/SIMD/default options, absent maps, replaced source arrays and the captured
Parakeet projection. Outputs must match the unmapped arithmetic bit for bit,
including nonzero-destination overwrite, immutable inputs and held-output
ownership. The auditor also compares every saved current/candidate output byte.
Normal and AVX512-disabled captured outputs must equal the retained graph output.

Each hardware-capable candidate process also runs 81 raw-kernel cases against
the unchanged compiled row-major routine: zero/small reduction sizes, all vector
and panel tail boundaries, dirty destinations, guard regions, special float
values with NaN payloads and the actual 640-by-8,198 weight. All output bits must
match. Guards and both original/packed inputs stay intact. Eight warmed calls
must allocate zero bytes. These are correctness cases for one reader, not a
performance search. Hardware-disabled processes exercise only public fallbacks.

Reuse the existing serial resource supervisor and verified transport. CPU 2 runs
builds/contracts; CPU 0 monitors. Require SDK 10.0.204, runtime 10.0.8, 2 GiB free
RAM and 1 GiB tmpfs before each job, less than 3 GiB owned RSS, at least 1 GiB
remaining RAM/tmpfs, at most 32 MiB job output and 64 MiB total resumed campaign files,
and 300 seconds/job. Restore only from the existing offline feed. Every .NET
build command disables terminal logging. Collection retains all source, binaries,
raw results and resource observations, excluding only regenerable HTTP cache.

The first campaign built Core and the consumer successfully, then failed before
inventory inspection because the bridge invocation supplied three arguments to
the qualified four-argument bridge. Its failure `6225233c` retains all 751
collected files and original tools. No numerical or performance process ran.
The fresh second namespace supplies the fourth argument and runs only the
inventory plus the six previously unstarted contract processes. It hardlinks
the exact compiled products and consumer; source, fixtures and arithmetic are
unchanged, and neither successful build is repeated.

The second campaign closes at failure `2c74d4ae`: inventory and both normal
processes pass, with 90 public and 81 guarded raw cases. The first AVX512-disabled
process stops at its hardware guard before arithmetic because the launcher used
`DOTNET_EnableAVX512F` instead of the existing lane's `DOTNET_EnableAVX512`.
The third namespace corrects only that launcher switch and executes the four
remaining processes. It reuses the same binaries and joins the retained normal
results during auditing; neither the builds, inventory nor normal calls repeat.

Local artifact: `artifacts/parakeet-decoder-packed-row-contracts-v3-amd-20260927`.
VM: `/dev/shm/lokad-decrow3-20260927`. Prefix commands below with
`C:/Python313/python.exe -X utf8 -B` from the repository root:

    tests/parakeet/decoder-packed-row/run.py prepare
    tests/parakeet/decoder-packed-row/run.py stage
    tests/parakeet/decoder-packed-row/run.py launch
    tests/parakeet/decoder-packed-row/run.py observe
    tests/parakeet/decoder-packed-row/run.py collect
    tests/parakeet/decoder-packed-row/audit.py

Observe the same live owner to terminal, then collect and audit once. Audit stdout
belongs outside the artifact. Retain any failed attempt; never repeat inference
to repair transport or reporting. Freeze tools before preparation.

No performance process is included here. After contracts pass, freeze one exact
captured-projection public MatMul comparison and its fallback controls. The
prospective gates in PLAN.md remain: at least 25% projection improvement, exact
outputs, no copy/scratch increase, same-product case means within 10%, unchanged
fallbacks within 5%; then at least 1% complete-corpus improvement with the existing
native/decision/ownership and application repeatability gates. A failed prediction
requires diagnosis, not a kernel sweep. Full model and release admission remain
necessary even if these contracts and the later operator screen pass.
