# Prepared recurrence: focused contracts

The actual candidate DLLs pass **150/150 cases in normal mode and 150/150 with
AVX512 disabled**, with no skipped tests. Core `3c23b44a` / Data `cc37b19e` match
the [independently reviewed build](build-20260924.md). Products were not rebuilt
for these tests. The selected release remains unchanged.

| Test class | Normal passes | AVX512-disabled passes |
|---|---:|---:|
| PreparedLstmWeightsTests | 22 | 22 |
| CpuExecutionProviderLstmTests | 24 | 24 |
| LstmInputBlockTests | 36 | 36 |
| LstmPanelAdmissionTests | 9 | 9 |
| LstmPanelOverflowRefusalTests | 1 | 1 |
| LstmOutputLaneTests | 10 | 10 |
| FoldBudgetTests | 14 | 14 |
| PackedWeightsTests | 16 | 16 |
| PackedWeightBudgetTests | 16 | 16 |
| IdentityTests | 1 | 1 |
| PreparedBudgetReceiptTests | 1 | 1 |

The 22 new source contracts cover atomic pair admission at zero/short/exact
budgets; combined matrix/convolution/recurrent accounting over repeated refresh;
immutable sources, independently held outputs, source replacement, shape and
consumer changes, aliases, offset/reversed views, invalidation/rebuild, and
scalar/direct/unsupported-geometry fallbacks. Deliberately poisoning internal
prepared values proves fresh execution contexts use them; scalar and direct
calls remain exact, and invalidation restores ordinary results.

A public-API-only preparation anchor runs from the same test assembly with the
selected Core `672e5f30` / Data `065b7a7f`. It fails specifically at the expected
13,107,200-byte preparation receipt versus the original zero; its product/runtime
identity test passes. The candidate passes that same anchor. Every test host
verifies actual product/consumer hashes and paths, one CPU2 processor,
.NET10.0.8 and its requested instruction mode.

All **50 resource observations** pass; peak owned RSS is **563,703,808 bytes**.
Every PID/birth owner is terminal. The full frozen census and all 302 test results
are in [machine-readable observations](contracts-20260924.json). Raw TRX files,
loaded identities, exact test sources and monitoring logs remain under
`artifacts/parakeet-prepared-recurrence-contracts-amd-20260924`.
Closure SHA-256: `aacf2dbe9265a96c5699155a86b1743462387990245ca8fdc6cc0e75cc73952a`.

[Prospective protocol](../prepared-recurrence-contracts-amd/README.md) fixes the
census and limits before execution. These contracts do not qualify the complete
model or establish a speedup. Actual decoder residency/dispatch, captured and
full native/public results, fixed complete-call/application comparisons, and
release regressions remain required before promotion or BENCHMARK.md changes.
