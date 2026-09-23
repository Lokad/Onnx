# Isolate short-projection packing from shared compilation changes

M43 demonstrated a full Parakeet application gain, but failed release admission
when the warmed e5-30 comparison regressed. A distinct candidate can retain the
short-projection mechanism while preserving the original general dispatcher and
every existing arithmetic kernel's body and compilation flags.

The admitted application comparison reduces the twenty-clip sum from 76.928211
to 74.463788 seconds (3.2035%). Two short clips account for the useful gain:
672-122797-0000 goes from 2.642254 to 1.470648 seconds, and 1221-135766-0002
from 3.497073 to 1.945496 seconds. That evidence motivates the guarded packing
route; it does not transfer performance admission to another implementation.

M43 changed more than the guarded route. It marked four shared float methods
for initial optimized compilation and moved the original general dispatcher
into a new method marked NoInlining. Source inspection confirms all four
arithmetic bodies and the relocated general body remain exact. The e5 failure
does not distinguish these composition changes from runtime history as causes.

The proposed scope is precise:

- Preserve `MathOps.cs` entirely. Add four internal, renamed copies of its
  float packer, two-row packed consumer, three-row packed consumer and unpacked
  odd-row kernel, each with first-use optimized compilation. Keep their bodies
  and ordered arithmetic exact; expose no new public API.
- Preserve the original `RunFloatMatMulKernel` body and flags in place. Redirect
  its four existing caller operands to a new small guarded dispatcher. The
  ordinary branch calls the unchanged original method.
- Keep M43's eligibility: 48–63 rows, reduction/output widths at least 1,024,
  at most 67,108,864 packed elements, SIMD/intrinsics/FMA enabled. Only that
  branch uses the new optimized copies. Preserve scratch rental/return, existing
  AVX512 switch behavior, two/three-row grouping and odd-row cleanup order.

The alternative adds private/internal implementation methods rather than
annotating shared existing kernels. It still changes call-site composition and
must earn fresh numerical, generated-code, component, complete application,
shared-model, root and package qualification. Preserve M43/M45 verdicts and all
padding failures. No unchanged scored retry or release-table change follows.

All 420 selected product inputs match their recorded identities. The inspection
also verifies M43's retained 421-file source, application closure and failed
release analysis. No new product snapshot or VM run has been prepared here.

[Exact source observations](source-observations-20260923.json) ·
[M43 application comparison](../first-use-kernels-app-amd/results-20260923.md) ·
[M45 release rejection](../../benchmarks/warmed-release-results/results-20260923.md)
