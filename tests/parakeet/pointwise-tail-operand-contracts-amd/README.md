# Qualify the diagnosed multiply-order repair

The first eight-row remainder candidate fails 260 exceptional AVX2-only cases.
The unchanged-product disassembly observation reproduces every result and shows
baseline B*A versus candidate A*B, with identical product+C addition order.
Reconstructing the first mismatching lane explains all 260 payload differences.

Reverse only the eight masked multiply operands in the isolated source helper.
Retain the same row sharing, full-panel code, complete-vector FMA, compact packing,
reduction order, guards, dispatch and fallbacks. Generated-code inspection must
confirm the correction in both modes; source spelling alone is insufficient.

Reuse the original build, consumer, cases, collector and auditor through a frozen
adapter. The consumer permits JIT disassembly in AVX512-disabled mode. Every raw
case still requires exact candidate/baseline bits within its hardware mode.
Both original scalar processes remain. Cross-mode equality is required for finite
cases; report exceptional differences separately because the unchanged baseline
already varies across modes. Preserve the original failed closure 58cb4026.

The adapter saves all generated tools and their hashes beside the new source.
From the repository root, prefix with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py prepare
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py stage
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py launch build
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py observe build
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py collect build
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py audit build
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py launch capture
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py observe capture
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py collect capture
    tests/parakeet/pointwise-tail-operand-contracts-amd/run.py audit capture

Each preparation, launch, collection and audit runs once. Observe the same owner
to terminal; preserve failure and diagnose it before another change. No model
execution or performance measurement is part of these numerical contracts.
