# Capture Winograd generated code through the qualified numerical consumer

This follows numerical closure
`749ecec8faabd8c46447362729393f4d8b35e085ffed4485438bbbf47bd0ac68` and reuses
the exact standalone binary `bb9cbee306d76f6f775cb64ca478133e438b84a456519cc4eb4f13228a16728b`.
Two fresh workers execute all87captured calls in256and512modes, retaining the
original native accuracy checks, every selected/candidate output hash and all
ownership checks. Neither source nor compiled consumer changes.

From repository root use `C:/Python313/python.exe -X utf8 -B` followed by
`tests/pyannote/winograd-input-codegen/run.py prepare`, then `stage`, `launch`,
`observe`, and terminal-only `collect`. Run `audit.py` after collection.
All workers must terminate before collection; no performance scoring is done.

The diagnostic filter is `*Winograd* Kernel256 Kernel512`. All emitted tiers
are retained. `DOTNET_JitStdOutFile` writes disassembly to each worker's
`jit.asm`, separate from managed progress messages. This release configuration
is declared in the [.NET10.0.8 source](https://github.com/dotnet/runtime/blob/v10.0.8/src/coreclr/jit/jitconfigvalues.h#L312).
No tiering, inlining or optimization override is used. The256worker disables
AVX512 solely for the required correctness/code-generation coverage.

Audit requires complete listings and optimized multiply/output bodies in both
widths. Independently inspect the new input transform: sixteen masked gathers per channel,
mask-copy costs, eight-lane ordered arithmetic, stack row storage, and division
only during batch coordinate setup. Keep all old non-input-transform bodies
for comparison. Inspect every optimized reduction: eight accumulators,
ascending channel order, weights reused across eight tile broadcasts, calls,
branches, spills, transform/inverse arithmetic and actual code size. Retain
unfavorable code and absent tiers explicitly. Numerical correctness is already
established; code inspection does not establish performance.

All numerical resource bounds remain:12GiB available/3GiB tmpfs preflight,
8GiB owned RSS,1GiB live available/tmpfs floors,900seconds/job,1GiB output and
2GiB artifacts. WorkersCPU2, supervisorCPU0; full pinned dependencies and
process identities. Only an independently reviewed capture permits preparing
the separate fixed complete-call screen described in the M31 plan.
