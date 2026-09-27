# The masked-tail failure is multiply operand order

The first eight-row remainder candidate passes 2,598 normal-mode cases but fails
260 exceptional AVX512-disabled cases. All finite cases and both 12-case public
scalar comparisons pass. Every first failure is in masked remainder columns.
No candidate performance has been measured; release scores remain unchanged.

A consumer-only diagnostic enables disassembly in the failing mode, preserving
baseline Core `47984318`, candidate `7cac6788`, all 2,598 cases and their order.
The sole consumer edit permits the additional declared flag. Every result,
including all 260 failures, reproduces exactly. The worker takes 4.543 seconds,
with nine resource samples and peak owned RSS 93,704,192 bytes.

| Operation | Compiled baseline | Candidate, AVX2-only |
|---|---|---|
| Multiply | B * A | A * B |
| Add | product + C | product + C |

Baseline `ymm3` contains B; `ymm4` and `ymm5` contain broadcast A values. It emits
`vmulps ymm4, ymm3, ymm4`. Candidate `ymm9` contains B and `ymm10` contains A;
it emits `vmulps ymm10, ymm10, ymm9`. All eight rows have that order. Normal mode
instead folds the A broadcast into the second operand and emits B * A.

An independent reconstruction using the original generator and observed operand
order reproduces **260/260 first mismatching payload pairs**. For M=64, reduction=64,
columns=33, output row 2, column 32 is `7fc12345` in baseline and `7fe12345` in
candidate. This identifies the cause of the observed failures; it is not a timing
result. The [.NET 10.0.8 intrinsic definitions](https://github.com/dotnet/runtime/blob/v10.0.8/src/coreclr/jit/hwintrinsiclistxarch.h#L633)
mark AVX addition and multiplication as commutative. Matching C# spelling alone
does not guarantee matching physical operands.

The selected repair reverses only the eight masked source multiply operands to
match baseline B * A. Inspect generated code and require all original within-mode
exact-bit checks before timing. Preserve row count, full panels, vector width,
reduction, FMA policy, dispatch, packing, allocation and ownership.

Separately, 482 cases passing both direct product comparisons have different
normal/AVX512-disabled hashes; all contain exceptional values, and 434 cannot enter
either new helper. Baseline NaN payloads therefore already vary across modes.
Require exact candidate/baseline bits within each mode and finite cross-mode
equality, recording exceptional cross-mode differences. This does not excuse any
of the 260 candidate failures.

Original failed closure `58cb4026` remains under
`artifacts/parakeet-pointwise-tail-contracts-amd-20260927`. Diagnostic evidence is
under `artifacts/parakeet-pointwise-tail-nan-diagnostic-20260927`: `review.json`,
`explanation.json` (`8cb34b8e`) and disassembly (`a294c853`). Build owner 1219717 /
birth1790528673.06 and capture owner 1220008 / birth1790528689.86 are terminal;
collections contain 33 and 41 files. Reproduction: [explain.py](../pointwise-tail-nan-diagnostic/explain.py).
Repair procedure: [README](../pointwise-tail-operand-contracts-amd/README.md).

## Source-order repair did not alter generated code

The repair is tested and ineffective. Candidate Core `1bbb6cff` changes only
those eight expressions and their comment relative to the first candidate.
Build review `e5fd1721` verifies the original scope and metadata requirements.
Both complete helper disassemblies are identical to their first-candidate
counterparts in each hardware mode: the JIT commutes the expressions back.

AVX512-disabled retains 260 failures. Normal has one exceptional failure at M=70,
reduction=63, columns=42, output row 0, column 2 (`7fe12345` versus `7fc12345`).
That case cannot enter either helper, and its first mismatch is in the untouched
full-panel arithmetic. Its cause is not established. All finite cases and both
scalar sets pass. No candidate performance has been measured.

Failed closure `1b83bf22` is preserved under
`artifacts/parakeet-pointwise-tail-operand-contracts-amd-20260927`. Build owner
1220532 / birth1790529199.53 and capture owner 1221077 / birth1790529248.33 are
terminal; collections contain 509 and 530 files.

Do not try further operand-spelling variants. The missing facts are the compiler
rule selecting physical operands and the baseline's within-mode NaN-payload
stability. Inspect pinned runtime source and isolate baseline behavior before
choosing a correction. Existing exact-bit verdicts remain failed; neither
candidate qualifies for promotion. Local allocated storage is 48.969 GB.
