# Capture Winograd generated code through the qualified numerical consumer

This follows numerical closure
`16eac8e386126ac62f8b00b52779cfcadc3a31dee986cc9d04e5cf14208d41a4` and reuses
the exact standalone binary `6eef05772e9b61bba18f4e5c045d920f2648f1f9b4551dd990d56f95e47eaf3f`.
Two fresh workers execute all87captured calls in256and512modes, retaining the
original native accuracy checks, every selected/candidate output hash and all
ownership checks. Neither source nor compiled consumer changes.

From repository root use `C:/Python313/python.exe -X utf8 -B` followed by
`tests/pyannote/winograd-contiguous-codegen/run.py prepare`, then `stage`, `launch`,
`observe`, and terminal-only `collect`. Run `audit.py` after collection.
All workers must terminate before collection; no performance scoring is done.

The diagnostic filter is `*Winograd* Kernel256 Kernel512`. All emitted tiers
are retained. `DOTNET_JitStdOutFile` writes disassembly to each worker's
`jit.asm`, separate from managed progress messages. This release configuration
is declared in the [.NET10.0.8 source](https://github.com/dotnet/runtime/blob/v10.0.8/src/coreclr/jit/jitconfigvalues.h#L312).
No tiering, inlining or optimization override is used. The256worker disables
AVX512 solely for the required correctness/code-generation coverage.

Audit requires complete listings and optimized multiply/output bodies in both
widths. Independently inspect the new contiguous helper: four row loads, four even/odd element permutations,
four half combinations, no gathers, ordered eight-lane arithmetic, stack row
storage and initialization. Require its actual optimized body in both widths.
Keep the masked border/tail body and all non-input-transform bodies for review. Inspect every optimized reduction: eight accumulators,
ascending channel order, weights reused across eight tile broadcasts, calls,
branches, spills, transform/inverse arithmetic and actual code size. Retain
unfavorable code and absent tiers explicitly. Numerical correctness is already
established; code inspection does not establish performance.

All numerical resource bounds remain:12GiB available/3GiB tmpfs preflight,
8GiB owned RSS,1GiB live available/tmpfs floors,900seconds/job,1GiB output and
2GiB artifacts. WorkersCPU2, supervisorCPU0; full pinned dependencies and
process identities. Only an independently reviewed capture permits preparing
the separate fixed complete-call screen described in the M32 plan.
