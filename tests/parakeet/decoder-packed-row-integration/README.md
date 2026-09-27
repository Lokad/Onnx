# Portable regression coverage for the fixed prepared-row change

Extract four xUnit facts from the closed contract consumer without changing the
candidate product. `PreparedSingleRowTests.cs.txt` preserves all 44 synthetic
public cases and 80 synthetic raw-kernel cases. Public tests cover dispatch width
boundaries, one/even/odd rows, shared-weight batches, explicit options, absent
mappings, replacement arrays, dirty destination overwrite, immutable inputs and
independent held outputs. Raw tests preserve reduction/vector/panel tails, guards,
NaN payloads, original and packed inputs, and zero warmed kernel allocations.

The raw facts skip on hosts without FMA. Both public facts remain portable to
hardware-disabled execution. Campaign file output, product/CPU identity gates and
the external real-model fixture stay in their original frozen evidence; they do
not become ordinary unit-test dependencies. The synthetic cases and reference
arithmetic are retained. Public copy/scratch counters now explicitly require zero,
which held for every original current/candidate call in all three modes.

`prepare_tests.py` reconstructs the fixture from the exact closed consumer and
checks every byte before creating the review artifact. It changes no root source,
builds no product, executes no test and downloads no model. The prepared receipt
`8b3a18b8ee3e22d87f30c030c4c4558823235b0178f2d42d8fadc17f29c7a035` binds the
four facts, candidate source stage, contract closure and unchanged source-policy
test. These are source checks, not proof of compilation or execution.

The artifact is already prepared at
`artifacts/parakeet-decoder-packed-row-integration-tests-20260927`. Reproduce
its review without creating or changing files:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-packed-row-integration/prepare_tests.py

Do not repeat `--prepare`. After all performance prerequisites are admitted,
actual-root qualification must apply the two fixed product paths plus this fixture,
prove all 3,284 Core / 697 Data methods match the measured candidate, and run the
entire existing test census plus these four facts. On the qualified AMD machine,
expected backend counts are 3,564 passed / 42 skipped normally and 3,474 / 132
with AVX512 disabled; tensors stay 394 / 0 in both modes. Preserve the independent
package consumer, dependency checks, warning census and all old test outcomes.
