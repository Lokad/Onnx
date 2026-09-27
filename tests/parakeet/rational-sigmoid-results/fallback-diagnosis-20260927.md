# Rational sigmoid: fallback evidence narrows the next decision

The additional observation used the exact qualified root and rejected rational
candidate, with no product change. The ordinary screen remains **rejected**:
74.992572% weighted reduction, 13 failed repeatability controls and four failed
fallback regression checks. No diagnostic clock below changes that verdict.

All four diagnostic processes completed. Every one of the original 143,520
clocks/counter records and every emitted method version is retained. The consumer
preserves the same 46 fixtures, values, calls, checks and rounds; allocation/GC
counters bracket the timed batch, with counter calls outside the timer.

The final optimized public wrapper is now observed in both products. Its fast
float and double loops each contain 14 instructions and four stack operands.
Both float loops increment pointers, call MathF.Exp once per element, and have
the same arithmetic/control opcode sequence. Register allocation differs: the
candidate reloads input and output pointers where current saves/reloads its loop
offset. Equal counts do not imply equal dependency latency. Double loops retain
the same operation/register sequence apart from stack locations and relocated
calls/constants. This evidence supplies no extra scalar arithmetic, call or bound
check to remove. It does not join a particular compiled version to every clock.

All 46 fixture/process combinations have equal minimum allocation bytes across
products. Empty calls always allocate 280 bytes; scalar calls always allocate
216 bytes. Sparse extra allocations in other fixtures are retained. Across all
rounds each process allocates approximately 42.54–42.55 GB, with hundreds of
generation-0/1 and over 160 generation-2 counter increments. The rational processes
complete the same census in about 13.7 seconds versus 36.1–36.4 seconds for current.
This establishes substantial allocation/collection activity in this synthetic
workload, not the complete application's allocation rate or allocator latency.

| Diagnostic observation | Complete mean | Exhaustive counter partition |
|---|---:|---|
| Candidate-2, empty | 130.247 ns/call | 179 batches without a collection-counter change average 79.470 ns; the one changed-counter batch averages 9,219.195 ns. |
| Candidate-1, scalar | 85.301 ns/call | 179 unchanged-counter batches average 65.790 ns; the one changed-counter batch averages 3,577.789 ns. |
| Current-0, empty | 94.915 ns/call | 179 unchanged-counter batches average 75.227 ns; one changed-counter batch averages 3,619.215 ns. |
| Current-3, empty | 111.744 ns/call | 178 unchanged-counter batches average 75.216 ns; two changed-counter batches average 3,362.676 ns. |

These partitions explain associations within the new observation. They keep every
batch and are **not corrected means or a new admission**. The counter interval is
slightly wider than the timer, and these events cannot identify historical events
in the failed ordinary screen. Empty unchanged-counter means are 75.227/75.216 ns
for current and 73.493/79.470 ns for candidate. There is no consistent large fixed
empty overhead in this observation.

Scalar-option complete means are 371.795/360.555 microseconds for current and
338.773/363.078 for candidate; the earlier ordinary-screen slowdown does not recur
here. Double calls remain slower: unchanged-counter means are 42.202/39.843
microseconds for current and 44.908/48.982 for candidate. Those double observations
leave runtime effects, allocation latency, code placement and actual tier execution
unresolved. They do not justify asserting universal fallback performance equality.

The next decision is **full-model numerical correctness of this exact candidate**.
The arithmetic hypothesis has substantial support and passed its focused numerical
checks; another coefficient, vector-width or unrolling trial would not resolve
the remaining model-accuracy question. Preserve the failed operator screen and
the unresolved double behavior. The living PLAN now permits correctness-only
model evaluation after this diagnosis; it does not retroactively admit the screen.

Use the complete retained Parakeet fixtures, both normal and AVX512-disabled
execution, original ORT scaled-error bound 1e-4, exact integer outputs, decoder
decisions/transcripts, immutable inputs and owned outputs. Floating-point tensors
must be checked numerically: the Pad campaign's extra bit-identity requirement
does not describe a new approximation. Record their differences from current
explicitly. Application timing needs its own prospective decision and original
gates after numerical correctness, before any source or BENCHMARK.md promotion.

[Every case/process counter partition and generated scalar loop](fallback-observations-20260927.json),
[read-only analyzer](analyze_fallback.py),
[failed ordinary screen](screen-20260927.md),
[focused numerical contracts](contracts-20260927.md),
[diagnostic protocol](../rational-sigmoid-fallback-diagnostic-amd/README.md).

Read-only reproduction:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/rational-sigmoid-results/analyze_fallback.py

The one-time publication is complete; do not use --publish again.
Build review: `37526eff84fa9238fa63c708b636f3e79527811301de7464c85ef73fb8c095ee`.
Diagnostic closure: `c6f0a34eb3467c69829ad958194f5f88375b83abb12fad8450c233bc41586049`.
Artifact: `artifacts/parakeet-rational-sigmoid-fallback-diagnostic-amd-20260927`.
All owners are terminal; do not rerun the campaign.
