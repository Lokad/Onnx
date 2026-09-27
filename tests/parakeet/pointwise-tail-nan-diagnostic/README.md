# Observe the failed masked-tail arithmetic

The original contract closure 58cb4026 remains failed: 260 exceptional-value
cases differ in NaN payload bits with AVX-512 disabled. Every first mismatch is
in the masked column remainder. Normal-mode disassembly does not identify the
instruction order in the failing mode.

This adapter builds only the same consumer with one reversible flag-validation
edit: permit the declared JIT disassembly flag alongside AVX512-disabled.
Reuse exact baseline Core 47984318 and candidate 7cac6788. Retain all 2,598 cases
in their original order; change no checks, arithmetic, product binary or JIT
tiering setting. Run one diagnostic process, with the existing bounded worker,
then compare its results to the original failed process and inspect the code.
This is neither a candidate variant nor a performance measurement.

From the repository root, prefix with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/pointwise-tail-nan-diagnostic/run.py prepare
    tests/parakeet/pointwise-tail-nan-diagnostic/run.py stage
    tests/parakeet/pointwise-tail-nan-diagnostic/run.py launch build
    tests/parakeet/pointwise-tail-nan-diagnostic/run.py observe build
    tests/parakeet/pointwise-tail-nan-diagnostic/run.py collect build
    tests/parakeet/pointwise-tail-nan-diagnostic/run.py launch capture
    tests/parakeet/pointwise-tail-nan-diagnostic/run.py observe capture
    tests/parakeet/pointwise-tail-nan-diagnostic/run.py collect capture
    tests/parakeet/pointwise-tail-nan-diagnostic/run.py review

Prepare/stage/launch/collect/review once; observe each same owner to terminal.
Preserve any differing diagnostic results. No product qualification or timing
experiment is allowed by this adapter. Inspect multiply and add operand order
in baseline and candidate masked loops before selecting a correction.
