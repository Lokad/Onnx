# Parakeet: explain the Microsoft ORT performance gap

The qualified release takes **73.82s versus ORT's 39.61s** for the twenty-clip
corpus: a **34.21s gap** and **1.864 ratio**. The next optimization will be chosen
from a matched diagnosis of these exact applications. The initial ORT capture
is running; results have not yet been collected.

The retained Lokad profile already constrains priorities. The entire
`ParakeetGeneration.Decode` call tree accounts for **12.529% and 12.548%** of
request samples in its two captures. That includes recurrent graph execution
and greedy decoding, not just the LSTM leaf. This suggests decoder-only work
cannot close most of the total gap. It is a sampled fraction with collection
overhead, not an exact phase clock or a promised saving. The rest must be measured
before assigning it entirely to the encoder. See the
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

The investigation will first compare frontend, encoder, decoder and remaining
time, then inspect the largest difference in the optimized graph and executed
native kernels. Source inspection identifies possible mechanisms; it does not
prove which implementation the installed wheel selected. Earlier intermediate
output instrumentation changed ORT's optimized graph, so the new diagnostic
preserves every original graph output.

Each subsequent candidate must name the observed discrepancy, its likely cause,
the maximum plausible whole-request benefit and one test that could reject it.
No diagnostic clock replaces the qualified table in [BENCHMARK.md](../../../BENCHMARK.md).

The in-flight decoder trial has now closed independently: **73.85s current,
70.35s candidate and 39.71s ORT**, a **4.74%** candidate reduction. All63
repeatability controls and21admission inequalities pass. The candidate remains
unintegrated pending shared-model and root/package checks; its1.772ratio is
not yet the release figure. [Complete application result](../prepared-recurrence-results/application-20260924.md).
