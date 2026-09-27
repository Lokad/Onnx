# Layout-only LSTM candidate: correctness diagnosis

The isolated candidate preserves all captured projection bits. Both SIMD modes
pass 176/176 LSTM tests. With hardware intrinsics disabled, 158 pass and 18 fail;
the separately executed qualified baseline reproduces all 18 failures exactly.
Each explicitly requests FMA, which the disabled-hardware configuration rejects
before prepared dispatch. This is an existing test portability assumption.

The [machine-readable diagnosis](contracts-diagnosis-20260927.json) binds the
original failed campaign and the separate baseline control. The original failure
`59f6f738` remains failed. Baseline control closure `63c97822` establishes 154
existing passes and 18 matching rejections, using the original baseline test DLL
and 116 verified runtime files, without a rebuild. No tests were removed or
rerun to replace the original verdict.

All four added facts pass in normal, AVX512-disabled and hardware-disabled modes.
All 760 captured projection hashes match across modes: 5,836,800 values in total.
The facts also cover indexing boundaries, exceptional float payloads, operand
order, guard regions, allocation and actual loaded product identities.

Compiled comparison confirms 3,282 unchanged Core methods, two changed methods
(packing and prepared dispatch), two added reader methods and all 697 unchanged
Data methods. Existing method flags, public declarations and assembly attributes
are unchanged. Core candidate is `ad97b4ad`; qualified control is `0d224bcf`.
The original validation method and the existing failing fixtures are unchanged.

This establishes correctness for the selected contracts, not performance or
complete-model qualification. Root product source and BENCHMARK.md remain at the
qualified release. Next is one fixed comparison of complete captured LSTM calls,
with an unchanged unprepared-path control; no arithmetic or layout variant sweep.
