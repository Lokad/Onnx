# The decoder executes the row-major kernel despite retaining packed weights

The qualified release's original Parakeet decoder executes `mm_m1_kblocked`
while its final projection weight already has a valid packed copy. This closes
the observation needed to select one addressing-only experiment. It does not
establish a new performance improvement or change the release comparison.

The workload is the first `english-16k` decoder step: A `[1,1,1,640]`, constant
B `[640,8198]`, original decoder graph and all four original outputs. Ordinary
control and traced execution each perform 256 warmups and 1,024 observations.
All **2,560 calls**, plus six initial control/capture calls, preserve exact
outputs, immutable inputs, held-output ownership and original graph structure.
Both processes retain the same mapping, source-array key and packed references.
Preparation retains 51,461,120 bytes within its 67,108,864-byte budget.

The worker stack contains `CPUExecutionProvider.MatMul`, `RunBatchedFloatMatMul`,
`RunFloatMatMulKernel` and `mm_m1_kblocked`. Original graph geometry and the
unchanged source guard identify the final projection as the decoder MatMul
eligible for that width-at-least-8,192 path; the other two MatMuls have width 640.
This combines sampled managed calls with graph/source evidence. The trace has
whole-decoder markers, not a separate marker or native argument capture per node.

| Sampled worker intervals within the 1,024 observed calls | Estimated milliseconds |
| --- | ---: |
| Row-major one-row kernel | 1,076.099 |
| Other MatMul kernels | 171.486 |
| Other work | 4,513.336 |
| **All intervals within calls** | **5,760.920** |

The row-major kernel represents **18.6793%** of these reconstructed intervals,
with intervals in 953 of the 1,024 calls. No sampled stack names the packed
final-row helper or packing routine. These durations are sampling estimates,
not direct kernel timers, sample counts, or complete-application fractions.
This repeats one real input, not the twenty-clip transcription workload.
All warmup intervals, outside intervals and other threads remain in the evidence.
No clocks are removed or corrected. Per-call JIT version is not established.

All 42,406 exported events reconcile, with 2,560 markers and zero lost events;
8,085 method/rundown records remain. Eight VM jobs pass, with 87 resource
observations and peak owned RSS 611,868,672 bytes. Whole-decoder counted copies
remain 10,240 bytes per call; neither those counters nor allocation totals are
per-node attribution.

## What this says about ORT and the next test

The [retained original-node comparison](../decoder-current-review/README.md)
puts final projection plus bias at 1.846170 seconds managed versus 0.718315 in
the dated ORT profile. Matching that cost is an approximately 1.128-second
opportunity, not a forecast. Its managed MatMul source is unchanged in this
qualified sigmoid release.

ORT's pinned `PrePack` / `BIsPacked` implementation and the actual node's
input profile support constant-weight preparation. Its byte-matched native
[one-row branch](../decoder-current-review/native-row1-20260927.md) uses packed
streams, but the retained native samples do not have a per-node caller join.
Do not assign their global sample weight to this projection. That missing
association does not change the decision to test consuming Lokad's existing
packed buffer, now that the managed bypass is observed.

The selected hypothesis is that the row-major reduction's 8,198-float stride
wastes locality compared with contiguous rows inside each existing 32-column
panel. This remains a causal hypothesis until tested. Keep the four AVX2
accumulators, B/A FMA operand order, ascending reduction, scalar tail, destination
semantics and ownership fixed; change only weight addressing and eligibility
for the existing prepared mapping. No wider-vector, unrolling, prefetch, fusion
or packing-format variants are part of this experiment. The current
`PackedFinalRowKernel` has different accumulation structure and is not a direct
substitute; see the [layout specification](../decoder-current-review/prepared-layout-20260927.md).

## Identities, reproduction and retained failures

Core: `65f15a41764660af6166c9a965943b6f28cac0c9505117d0fd4beaa2687f9d03`.
Closure: `ef94387e0517b08fd87d010c08d99e0b801e1bd7d2809c04b1dcec614333397d`.
Artifact: `artifacts/parakeet-decoder-projection-observation-v3-amd-20260927`.
Supervisor 1165080 / birth 1790484146.34 is terminal, exit zero.
The [machine-readable summary](observations-20260927.json) records all product,
consumer, mapping, sample and audit identities. Raw data stays in the artifact.

The first observer failed compilation; the second passed ordinary controls but
failed before tracing because of the diagnostic socket path length. Both remain
frozen. The third uses a shorter VM path with the same observer and controls.
Its first local audit invocation lacked `psutil`; the next encountered Windows
separators while reconstructing Linux command arguments. The final invocation
uses the unchanged auditor, forbids all local process-monitor calls, and gives
only remote command paths POSIX semantics. All real process observations were
made on Linux. Both errors and every invocation are retained and hashed in the
summary. No VM workload was repeated to repair local interpretation.

Reproduce the published summary, read-only, from the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-projection-observation-results/summarize.py

Publication with `--publish` is complete. Never replay this VM campaign.
