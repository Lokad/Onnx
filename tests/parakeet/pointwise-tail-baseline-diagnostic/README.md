# Test the exact numerical baseline against itself

Use qualified Core 47984318 in both assembly load contexts and the exact compiled
consumer from the preceding disassembly diagnostic. No product or consumer is
rebuilt. Keep all 2,598 cases, their original order, numerical/ownership/guard
checks, runtime, affinity and bounded worker. Run normal and AVX512-disabled once,
with declared disassembly. This observes baseline stability under the existing
consumer; it cannot qualify a candidate or establish universal JIT stability.

From the repository root, prefix with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/pointwise-tail-baseline-diagnostic/run.py prepare
    tests/parakeet/pointwise-tail-baseline-diagnostic/run.py stage
    tests/parakeet/pointwise-tail-baseline-diagnostic/run.py launch capture
    tests/parakeet/pointwise-tail-baseline-diagnostic/run.py observe capture
    tests/parakeet/pointwise-tail-baseline-diagnostic/run.py collect capture
    tests/parakeet/pointwise-tail-baseline-diagnostic/run.py review

Prepare, stage, launch, collect and review once. Observe the same live owner to
terminal. Retain every failure and the generated code; compare these facts with
the already failed candidate campaigns before choosing any correction.
