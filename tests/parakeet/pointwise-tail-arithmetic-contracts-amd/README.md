# Qualify arithmetic after the baseline NaN diagnosis

The identical-product baseline check fails strict NaN-payload equality in 214
normal and 218 AVX2-only cases. It even reproduces the candidate campaign's
first normal failure outside either helper. Keep those failures and both failed
candidate closures. They establish that exact arithmetic NaN payloads are not
stable in the existing consumer, even without a product change.

This new contract requires exact bits for every non-NaN value, including signed
zero, subnormals and signed infinities; both sides must be NaN for a NaN output.
Every payload difference is counted separately and `bit_exact` is false for any
affected raw case. Original input/packed-buffer byte hashes, guards, three
accumulations, zero warmed-call allocation and independent oracle remain intact.
All 2,598 cases and their original order remain. Nine comparator controls include
finite-bit, zero-sign, infinity-sign, NaN-versus-finite and length rejections.

Reuse original candidate Core 7cac6788 and baseline 47984318. Do not carry forward
the ineffective source-order edit or rebuild a product. Build only this consumer,
then run identical-baseline normal/AVX2, candidate normal/AVX2, and both public
hardware-disabled modes. Keep disassembly for all four raw processes. All finite
case records must also match across those processes. Existing repository tests,
pure-copy payload guarantees and complete-model numerical gates remain unchanged.

From the repository root, prefix with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/test_audit.py
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/run.py prepare
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/run.py stage
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/run.py launch build
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/run.py observe build
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/run.py collect build
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/audit.py build
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/run.py launch capture
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/run.py observe capture
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/run.py collect capture
    tests/parakeet/pointwise-tail-arithmetic-contracts-amd/audit.py capture

Prepare/stage/launch/collect/audit once, observe the same owner to terminal, and
preserve every failure. This qualifies arithmetic only. Review generated code,
then test the original remainder-sensitive performance prediction and complete
Parakeet admission. No timing or release claim follows merely from these checks.
