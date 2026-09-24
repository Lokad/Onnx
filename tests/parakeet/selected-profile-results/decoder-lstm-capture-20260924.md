# Actual Parakeet decoder LSTM capture

Both recurrent nodes are now captured and independently qualified against
Microsoft ORT on every one of the selected trajectory's 190 decoder steps.
The selected Core `672e5f30` and Data `065b7a7f` remain unchanged. This is a
correctness fixture bank for the prepared-recurrence trial, with no speed claim.

| Case | Decoder steps | Complete LSTM calls | Native output arrays, two repeats | Largest scaled error |
|---|---:|---:|---:|---:|
| english-16k | 37 | 74 | 444 | 9.1791153e-06 |
| french-44k-stereo | 29 | 58 | 348 | 6.91413879e-06 |
| jfk-48k-stereo | 46 | 92 | 552 | 1.28746033e-05 |
| english-token-limit | 4 | 8 | 48 | 5.6931749e-06 |
| english-frame-limit | 37 | 74 | 444 | 9.1791153e-06 |
| silence | 0 | 0 | 0 | 0 |
| english-repeat | 37 | 74 | 444 | 9.1791153e-06 |

Each nonempty case ran twice normally and twice with additional per-execution
output bindings. All **760 original decoder executions / 3,040 output arrays**
matched the prior selected results exactly, including token and duration
decisions. Each pass carried its own accepted recurrent states. All original
optimized node fields/attributes, output names, initializer references/shapes/
contents and held outputs remained unchanged. No original model was edited.

Two tiny native graphs copy the original LSTM nodes byte for byte, preserve
opsets/IR, and feed captured weights and states at the actual one-step, batch-one,
input/hidden-size-640 geometry. ORT **1.29.0 CPUExecutionProvider** runs with one
intra/inter thread, sequential execution, all optimizations and no spinning.
All **2,280 native arrays / 1,459,200 values** pass
`abs(managed-native)/max(1,abs(native)) <= 1e-4`. The largest error is
**1.28746032715e-05**. Native repeats are exact; inputs and
held outputs remain unchanged. Fed-weight graphs are correctness instruments;
they do not establish an ORT performance baseline.

Capture tensors occupy **27,374,080 bytes in 443 unique files**, below the frozen
64 MiB cap. Every W/R/B constant matches the original model's raw bytes. No encoder
execution, model download or full model copy was needed. All 115 resource samples
pass; peak owned RSS is **754,020,352 bytes**. Every worker is terminal.

[Prospective protocol and commands](../decoder-lstm-capture-amd/README.md) fix
all cases, resource bounds and correctness controls before execution.
[Machine-readable observations](decoder-lstm-capture-20260924.json) retain case
counts, both node descriptors, constant hashes, loaded native library identities
and resource summaries. All raw captured/native arrays, control rows and logs
remain under `artifacts/parakeet-decoder-lstm-capture-amd-20260924`.
Closure SHA-256: `28c7afe448ed16e3bb19d29232c3f90eb2d72c096196ae27792f5261afa5b64f`.

Next: implement the isolated cache within the existing aggregate 64 MiB decoder
budget, preserve the ordered projection arithmetic, prove lifecycle/dispatch,
then qualify complete trajectories and fixed performance comparisons. The
release and BENCHMARK.md change only after application and regression admission.
