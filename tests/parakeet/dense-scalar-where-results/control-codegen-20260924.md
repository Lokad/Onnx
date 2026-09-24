# Exact-consumer Where code-generation diagnostic

**Diagnostic passed; no performance admission.** The four fresh processes reuse the rejected control’s exact consumer and identical selected release binaries. The only extra managed setting is the declared disassembly flag. No build or product change occurred.

Whole-method optimized code sizes in process order:

| Method | Outer 0 bytes | Middle 1 bytes | Middle 2 bytes | Outer 3 bytes |
|---|---:|---:|---:|---:|
| Public provider Where | 2265 | 2265 | 2265 | 2265 |
| Generic float Where | 2887 | 2893 | 2895 | 2898 |
| Consumer Work<float>.Phase | 6939 | 6939 | 6939 | 6939 |

The generic float entries all retain 23 profiled inlinees, no external BroadcastShape calls and four integer divisions. Their branch layouts differ, including placement of the dimension-zero branch and selection-loop blocks. Equal inline counts do not imply equal generated code. The provider retains calls for all nine supported element types. The float consumer retains the public provider call inside its timer; it does not invoke the candidate helper.

The emitted whole-method Tier1 census is:

| Element type | Generic Where sizes, processes 0 / 1 / 2 / 3 | Consumer Phase whole Tier1 |
|---|---|---|
| float | 2887 / 2893 / 2895 / 2898 | 6939 / 6939 / 6939 / 6939 |
| long | 2401 / 2410 / 2407 / 2396 | absent / absent / absent / absent |
| bool | 2403 / 2397 / 2398 / 2406 | absent / absent / absent / absent |
| byte | absent / absent / absent / absent | absent / absent / absent / absent |
| int | absent / absent / absent / absent | absent / absent / absent / absent |
| uint | absent / absent / absent / absent | absent / absent / absent / absent |
| ulong | absent / absent / absent / absent | absent / absent / absent / absent |
| double | absent / absent / absent / absent | absent / absent / absent / absent |
| System.Half | absent / absent / absent / absent | absent / absent / absent / absent |

“Absent” means no such complete body appears in these retained listings; it does not mean the calls were skipped. Every case completed. The other consumer specializations emit instrumented Tier0 and Tier1-OSR bodies. A phase can execute many internal iterations while its method is entered only twice per case. Here 210 cases are float, three are Int64, and each of the other seven types has one case. Increasing the internal sample count does not increase the number of phase-method entries.

On-stack replacement (OSR) enters optimized code during an already-running invocation. A whole-method Tier1 entry is a separate compilation. The [.NET runtime explanation](https://github.com/dotnet/runtime/blob/v10.0.8/docs/design/features/OsrDetailsAndDebugging.md) describes that distinction and the effect of internal loops on benchmarking. The versioned document is retained with its hash under `artifacts/parakeet-where-control-runtime-source-20260924`.

These observations motivate a distinct consumer with one shared, non-generic batch-timing method called for every sample and round-robin warmup across all 220 cases. That is a hypothesis to test, not a proven fix. Keep the original 600 warmup / 180 measured samples, complete public calls, deterministic batches, all cases and unchanged acceptance limits. Qualify its generated code and require a new identical-binary control before any candidate race.

The diagnostics do not establish the exact tiers of the original untraced control, attribute its false 37.6% improvement to a particular mechanism, or admit M59 performance. Branch layout, the consumer and lower-level callees remain distinct possible influences. Do not repeat the unchanged rejected control, trim clocks, drop cases or relax gates.

All five jobs pass: 2,256 resource observations, 880 setups, 686,400 retained clocks and 232,929,840 complete public calls (53,753,040 within measured intervals). Peak process-tree RSS is 401,752,064 bytes. All 880 setups match the prior control’s input hashes, shapes and qualified output hashes. Exact result bits, metadata, unchanged stores and held/output ownership pass.

All 192 emitted bodies are retained to their terminal byte count, with every named branch target resolved. The [complete body census, call lists and comparison identities](control-codegen-review-20260924.json) identify the full listings and diffs. Normalization removes only explicit relocation fields; unmarked pointer-like constants, registers, instructions and branches remain. This avoids hiding code differences, but means address changes can still appear in the diffs.

Closure: `9b4ad473ff373bea1b50ecb20d9b3f7896fb8b84dbbfb0c71c70692f55290e1a`. Tools frozen at `09b16e50`. Supervisor 899476 / birth 1790219844.43 and all descendants are terminal. Consumer `19f25657` (52,736 bytes), Core `672e5f30`, Data `065b7a7f`. Full collection: `artifacts/parakeet-dense-scalar-where-control-codegen-amd-20260924`.

The selected release and BENCHMARK.md remain unchanged: Parakeet / ORT 1.864, Pyannote / ORT 1.154. M59 is numerically qualified and still unmeasured.
