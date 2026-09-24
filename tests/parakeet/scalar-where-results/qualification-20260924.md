# Uniform-mask qualification and Boolean compatibility correction

V2 passes its fixed canonical-mask census: four workers,123cases each and
12,220,016output values in total. Selected/candidate outputs are bit-identical
in both normal and AVX512-disabled modes. Every independent coordinate check,
captured NumPy bit selection, input/guard invariant, mutation, held output,
output ownership and expected exception check passes. The candidate helper
admits44cases per mode. All67resource samples pass; peak RSS329,359,360bytes.
Owner885012/birth1790208278.15 and descendants are terminal, code0.

Two diagnostic code-generation processes exercise32cases81times each. All ten
emitted bodies parse with resolved labels. The new1259-byte FullOpts helper
checks types/ranks/shapes, handles empty output before reading the mask, scans
once and dispatches to Fill or Memmove into fresh storage. Its body contains
no coordinate division or float arithmetic. The runtime library's scan/fill/copy
bodies were not captured. Tier inventories differ between processes, so this
does not establish which tier runs during future timings. Full codegen evidence
is codegen-review-20260924.json (5607f929).

That review exposed an untested compatibility case. The candidate searched for
the exact opposite Boolean byte. Eight separately frozen raw-byte masks show
that V2 mishandles[0,2]and[0,255]in both modes: with scalar true value-10000
and false values[10,20], selected Where returns[10,-10000], but V2 returns[10,20].
The other six masks match. Selected passes all eight. Inputs remain unchanged.
The independent NumPy oracle selects by byte!=0 and confirms the discrepancy.
All seven probe jobs and23resource samples pass; peak RSS323,854,336bytes.
Owner886212/birth1790208956.28 is terminal. This is a successful diagnostic
finding, not numerical admission of the candidate.

ONNX's generated schema requires0/1Boolean encodings. This probe does not relax
that format rule; it protects existing public tensor behavior when raw Boolean
storage is supplied. DenseTensor<bool>'s memory constructor and the current raw
Boolean decoder preserve those bytes. V2 remains unfit for integration or timing,
despite its passing canonical-mask census. No M56 performance trial has run.

V3 preserves the same scope and replaces exact opposite-byte search with zero/
nonzero testing: IndexOf(0) for a true-leading mask, IndexOfAnyExcept(0) for a
false-leading mask. Source6971271e builds at8f5958cb. All3,188otherCore and697Data
methods/flags/API remain exact; the existing Where body is preserved after the
same14-instruction guard and one local. The new helper has193IL instructions.
Four build jobs/74resource samples pass, peak RSS505,008,128bytes; owner886848/
birth1790209124.21 is terminal. Candidate Core70275a50/Data7d26bf8c.

The expanded131-case V3 census retains all123original cases and adds every raw
mask. Its numerical/codegen admission is separate and must precede a fixed
component screen. The selected root remains source81f75c38,Core672e5f30/
Data065b7a7f. BENCHMARK.md continues to describe that qualified release.

| Evidence | Closure SHA-256 |
| --- | --- |
| V2 canonical numerical census | a5f2bfda923acea5673607935b5487878211148bef88579cb490c1409539d55a |
| Raw Boolean diagnostic | e83bf8c65c0d695328e68f77a9ad8f664462553973cdcd8187edcd94792282c6 |
| Corrected V3 build | 8f5958cb468b3a35582150de113cdaec61bb4389afea98ca3a7c40fa67071909 |

qualification-observations-20260924.json retains the complete analysis records,
all diagnostic outputs and exact product/consumer identities. Frozen tools:
canonical numerics a5f1baca,raw diagnostic9623aecd,V3source/build da3e8fdc.
