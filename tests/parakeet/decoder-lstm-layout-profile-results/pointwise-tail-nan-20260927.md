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

## Identical baseline binaries also fail payload equality

The subsequent baseline-only check uses exact Core `47984318` in both assembly
load contexts, without compiling a product or consumer. It retains all 2,598
cases and their order. It reports **214 normal and 218 AVX2-only failures**.
Every first mismatch is NaN versus NaN in complete-vector columns. Respectively
86 and 90 cases cannot enter the proposed helpers. The first normal failure is
exactly the previous repair campaign's M=70, reduction=63, columns=42, row 0,
column 2. Thus the demanded within-mode NaN-payload identity is not stable even
for the baseline under the existing consumer. All finite cases pass.

Evidence: `artifacts/parakeet-pointwise-tail-baseline-diagnostic-20260927`,
review `a199e092`, 28 collected files, terminal owner 1221768 / birth1790529827.84.
The two processes take 5.035 and 5.036 seconds with ten resource samples each.
These are diagnostic durations. This result does not itself qualify a candidate.

The installed runtime revision `94ea82652cdd4e0f8046b5bd5becbd11461482ca` belongs
to dotnet/dotnet, the combined .NET source repository. Its
[source manifest](https://github.com/dotnet/dotnet/blob/94ea82652cdd4e0f8046b5bd5becbd11461482ca/src/source-manifest.json)
maps runtime to `b82454cad0aaaae3db2cf18fbf2cccc36e201ccc`, matching tag v10.0.8.
Installed JIT hash is `3cb5bbf7`; exact source files and hashes are retained under
`artifacts/parakeet-pointwise-tail-jit-review-20260927`.

The applicable [lowering rule](https://github.com/dotnet/runtime/blob/b82454cad0aaaae3db2cf18fbf2cccc36e201ccc/src/coreclr/jit/lowerxarch.cpp#L9869)
can swap commutative operands to put a memory operand second. Otherwise it uses
a [register-allocation preference](https://github.com/dotnet/runtime/blob/b82454cad0aaaae3db2cf18fbf2cccc36e201ccc/src/coreclr/jit/lowerxarch.cpp#L7561)
that favors a local over an expression temporary for that position. This explains
why spelling B*A does not force that physical order: B is shared while A is a
broadcast expression. Normal execution can instead fold A's broadcast into the
second operand. Read-modify-write register swapping is a separate rule; the
ordinary VEX AVX multiply does not take that rule. No further spelling trial is
needed. Exact generated code remains the direct evidence for the observed order.

The corrected arithmetic contract therefore requires **exact non-NaN bits and
matching NaN classification**, counting payload differences separately and marking
affected raw cases `bit_exact=false`. This preserves signed zero, subnormals,
signed infinities, every original case, immutable input/packed bytes, guards,
allocation and independent oracle. It also allows all exceptional cases to reach
their remaining output and ownership checks instead of stopping at the first
payload difference. Existing repository tests, copy-bit guarantees and complete
model error/finiteness/transcript gates remain unchanged.

Keep both previous candidate closures failed. The new
[arithmetic qualification](../pointwise-tail-arithmetic-contracts-amd/README.md)
uses original candidate `7cac6788`, baseline A/A controls and both public scalar
modes. The ineffective source-order edit is not carried forward. No performance
claim or release promotion follows until numerical and application gates pass.

## Arithmetic qualification passes; performance remains unmeasured

The corrected contract passes all **10,392 raw cases and 24 scalar cases**.
Every non-NaN bit, NaN classification, input/packed-byte identity, guard and
allocation check passes. Nine comparator controls run in each process; three
local auditor tests reject numerical, classification, ownership, coverage and
false bit-identity claims. All finite case records agree across the four raw
processes. Products remain original Core `47984318` and candidate `7cac6788`.

In this new consumer, baseline A/A records no payload differences; the earlier
214/218 failures remain evidence that the strict payload gate is unstable.
Candidate normal records 198 differing payload comparisons in one case;
AVX2-only records 14,097 in 274 cases. Counts span all three comparisons per case.
These cases are explicitly marked `bit_exact=false`; their complete remaining
outputs and ownership checks now run and pass.

Closure `558f2a52` and generated-code review `3715408c` are retained under
`artifacts/parakeet-pointwise-tail-arithmetic-contracts-amd-20260927`.
Build/capture owners 1222396 / birth1790530359.63 and 1222694 / birth1790530385.84
are terminal; collections contain 48 and 76 files. All four raw processes take
about 5.05 seconds, with ten resource samples each; peak owned RSS is 93,544,448
bytes. Scalar processes take about 0.51 seconds each. These are diagnostic clocks.

Both helpers keep eight independent accumulators without vector stack spills.
Complete vectors use eight FMA instructions per reduction step; masked vectors
use eight multiplies and eight separate adds. Their complete generated code
matches the original candidate in each mode. Normal/AVX2-only sizes are 462/494
bytes for the full-vector helper and 788/828 for the masked helper.

Numerical qualification now permits the fixed shape experiment in PLAN.md.
No speedup is measured yet. Parakeet's qualified release remains 46.931723 seconds
versus ORT 39.513180 seconds, ratio 1.188.
