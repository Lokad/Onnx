# ORT Parakeet padding obtains no new arena space in measured calls

The existing exact-model ORT profile already contains the allocator counters
needed to narrow this investigation. All **2,880 measured encoder Pad calls**
have zero change in both requested live bytes and arena live bytes, and zero
growth of the memory held by the arena. This is observed execution evidence,
not an inference from a default setting. No new inference or VM run was needed.

| Phase / family | Calls | Calls with zero requested-byte delta | Output payload processed | Arena growth inside Pad |
|---|---:|---:|---:|---:|
| Warmup attention | 480 | 0 | 620,545,536 B | 0 B |
| Warmup convolution | 480 | 0 | 278,888,448 B | 0 B |
| Measured attention | 1,440 | 1,440 | 1,861,636,608 B | 0 B |
| Measured convolution | 1,440 | 1,440 | 836,665,344 B | 0 B |

For every warmup Pad call, the requested live-byte delta equals its complete
output payload. The arena already holds enough storage: its held-byte delta
is zero in those calls too. In the three measured passes, the same nodes
produce their outputs without increasing live arena space at the node boundary.
The measured Pad durations sum to 150,525 us over three corpora, reproducing
the retained 0.050175-second per-corpus attribution.

The reader accounts for all 80 encoder runs, all 48 distinct Pad nodes per run,
the exact observed attention/convolution shapes and all 3,840 calls. There are
twenty distinct encoder-feed shapes and twenty distinct keys computed by the
pinned `CalculateMemoryPatternsKey`. Nineteen different encoder output frame
counts do not mean nineteen input shapes: two clips produce 156 encoder frames
from different frontend lengths. Each feed shape first occurs in warmup and
recurs exactly in every measured pass.

The retained adapter enables sequential CPU execution with one intra/inter-op
thread and all graph optimizations. It does not override memory patterns,
memory reuse or the CPU arena; their defaults are true in this exact ORT source.
The pinned execution frame records a pattern on an initial run and can obtain
subsequent internal tensors from matching blocks of a larger buffer. That
mechanism explains the observed transition. Planned buffer reuse is another
source-supported route. Neither branch instruction addresses nor individual
buffer addresses were captured, so their exact split is not established.

The counter semantics were checked in `sequential_executor.cc` and
`bfc_arena.cc`: requested/live counters describe allocations currently held;
the held-byte counter describes storage acquired by the arena. A zero delta
alone cannot exclude an allocation followed by a matching free within a node.
Here, the reviewed constant Pad implementation obtains one nonempty output,
copies inner blocks and fills borders, without an arena-backed temporary that
could explain a matching free. This supports preobtained output storage, while
leaving its exact address and allocation-plan provenance unobserved.

Do not run the optional node-memory-stat collector merely to repeat this
evidence. Its source accumulates maxima across runs, so first-run allocations
would obscure the transition of interest. Changing memory-pattern settings
would also change the workload whose behavior we are trying to explain.

The current Lokad copy candidate creates an independently owned managed array
for every public Pad. This differs materially from the observed warmed ORT
node lifecycle. It does not by itself prove why the candidate's two processes
have different timings, nor justify tensor pooling that could violate held-output
ownership. The next decisive observation is per-call thread CPU, page faults,
context switches, allocation and collection counters on the unchanged current
binaries. The inner public-Pad timer stays intact; the wider counter bracket
and an empty-bracket calibration expose instrumentation cost. No new score,
duration sweep or allocator implementation follows from this report alone.

[All observed calls](ort-pad-allocations-20260926.csv),
[verified inputs and per-request totals](ort-allocation-observations-20260926.json),
[reader](inspect_ort.py). The reader verifies the original closed receipt,
collection, raw profile, adapter and manifest, and hashes seven pinned ORT
source files. Original profile closure: `615353b4d8524f8935076357c0949859553c27826d50d53ebd82fb002d146a85`.
The release comparison remains 53.107381 versus 39.207695 seconds.
