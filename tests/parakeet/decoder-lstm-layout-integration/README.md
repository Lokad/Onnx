# Portable tests for the selected LSTM weight layout

Extract the two raw projection facts from the frozen LstmLayoutContracts.cs.
Keep both fact bodies and their Hash, Equal, Values, Pack and Raw helpers
unchanged. Remove only unused capture/VM helpers and imports, and rename the
fixture to PreparedLstmProjectionTests. The 61 synthetic cases preserve exact
float bits, NaN payloads, operand order, boundary guards, immutable inputs and
zero allocation for eight repeat calls. These facts need no external model.

The original four facts passed normally, with AVX512 disabled and with hardware
intrinsics disabled. The original campaign remains failed because 18 existing
explicit-FMA requests reject in scalar mode. The separate qualified-baseline
control reproduces those same 18 failures; its diagnostic closure is 63c97822.
All captured projection hashes and exact-VM identity checks remain in the closed
campaign. Existing PreparedLstmWeightsTests still cover graph preparation,
admission, scalar bypass, stale weights, ownership and shared budget.

This directory does not apply root source or compile a product. The extracted
fixture must later pass both complete root test suites before release promotion.
Freeze these files, then run once from the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/decoder-lstm-layout-integration/prepare_tests.py --prepare

The receipt and exact source diff go to
artifacts/parakeet-decoder-lstm-layout-integration-tests-20260927.
Without --prepare, the command verifies that receipt without mutating it.
