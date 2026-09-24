# Parakeet: explain the Microsoft ORT performance gap

The qualified release takes **73.82s versus ORT's 39.61s** for the twenty-clip
corpus: a **34.21s gap** and **1.864 ratio**. The ORT diagnosis is complete:
original application phase clocks, optimized graphs and actual native dispatch
are now reconciled. [Matched Lokad phase/operator clocks](../managed-phase-results/results-20260924.md)
are also complete: encoder **63.840s versus 37.573s**, decoder **9.044s versus
1.905s**. The 217 constant-projection groups explain **8.067s** of the **26.267s**
encoder difference, about 31%.

| ORT phase, original twenty-clip corpus | Seconds |
|---|---:|
| Encoder | 37.573 |
| Decoder | 1.905 |
| Frontend | 0.228 |
| Outside graph calls | 0.111 |
| **Complete request** | **39.817** |

The encoder's **217 constant-weight projections consume 29.44 profiled seconds**;
its 72 dynamic matrix products consume only **0.83 seconds**. The 1,993 encoder
nodes, 23 decoder nodes and 35 frontend nodes match the executed profile exactly.
All 48 scaled projections fuse multiplication by 0.5. Runtime input observations
and the matching source support preparation of constant B weights.

Native samples identify **`MlasGemmFloatKernelAvx512F` at 90.87% of measured sample
weight**. All **8,904 bytes** of the function match assembly built from the exact
Microsoft revision reported by the installed wheel. Sampled hot instructions
include its 12-row × 32-column AVX-512 block. This function also serves other
matrix work; its entire share cannot be assigned to constant encoder projections.
See [the graph and kernel findings](ort-kernels-20260924.md) and
[complete phase/operator clocks](ort-phases-20260924.md).

Lokad already has a 12-row AVX-512 kernel, enabled for eligible prepared weights.
Its per-call packing path uses two-/three-row kernels by default, and its encoder
retains 37 weights at the 256 MiB cap. The complete matching projections,
including fused scale, cost **37.509s versus 29.442s**. Their measured excess
does not justify assigning the whole encoder gap to matrix dispatch or packing.
Prior rejected toggle and cache trials remain closed.

The next bounded hypothesis is attention slice materialization. All 24 matched
Slice→Reshape pairs cost **2.962s versus 0.024s**. Selected Lokad source copies
sliced tensors through per-element coordinate translation; the matching ORT
source uses contiguous slice copies and permits Reshape aliasing. Confirm the
actual managed layout, then test one guarded bulk-copy change. The observed
ceiling is about **2.94s, or 4% of the whole request**, not a promised saving.
See the [matched pair evidence and rejection conditions](../managed-phase-results/results-20260924.md).

The retained Lokad profile already constrains priorities. The entire
`ParakeetGeneration.Decode` call tree accounts for **12.529% and 12.548%** of
request samples in its two captures. That includes recurrent graph execution
and greedy decoding, not just the LSTM leaf. This suggests decoder-only work
cannot close most of the total gap. It is a sampled fraction with collection
overhead, not an exact phase clock or a promised saving. The new exact phase
clocks confirm that the encoder contains most of the difference. See the
[original stack weights](../selected-profile-results/stacks-20260924.csv) and
[profile scope and overhead](../selected-profile-results/results-20260924.md).

The native workload uses the original frontend, encoder and decoder graphs,
ORT 1.29.0 CPUExecutionProvider, one intra/inter-op thread, sequential execution,
all graph optimizations and no spinning. Each corpus pass contains twenty frontend
calls, twenty encoder calls and **1,200 decoder calls**. The new
[diagnostic](../ort-diagnosis-amd/README.md) retains the original application
consumer and correctness checks, adding phase clocks, runtime shapes and ORT's
built-in operator profile. An unprofiled diagnostic control measures observation
overhead before interpreting the profile.

The phase observer adds 0.28% wall time relative to the preceding native run;
the ORT profiler adds 1.50% relative to that observer, and native sampling adds
0.61%. These separate processes also include ordinary timing variation. All
original request checks pass. All 32,426 raw samples reconcile, with no lost
samples. No overhead is subtracted, and no diagnostic result is a new benchmark.
Earlier intermediate-output instrumentation changed ORT's optimized graph;
this diagnosis preserves every original graph output.

Each subsequent candidate must name the observed discrepancy, its likely cause,
the maximum plausible whole-request benefit and one test that could reject it.
No diagnostic clock replaces the qualified table in [BENCHMARK.md](../../../BENCHMARK.md).

The preceding decoder trial closed independently: **73.85s current,
70.35s candidate and 39.71s ORT**, a **4.74%** candidate reduction. All 63
repeatability controls and 21 admission inequalities pass. The candidate remains
unintegrated pending shared-model and root/package checks; its 1.772 ratio is
not yet the release figure. [Complete application result](../prepared-recurrence-results/application-20260924.md).
