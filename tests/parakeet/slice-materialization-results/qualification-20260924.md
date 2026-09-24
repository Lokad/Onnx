# Guarded slice materialization: copying contracts pass

The isolated candidate passes **369/369 tensor tests with AVX-512 enabled and
369/369 with it disabled**, with no skips. Both modes consume the same Core and
test binaries. An explicit test checks the loaded Core hash, one visible
processor and the actual `Avx512F.IsSupported` value. Full-model numerical and
performance qualification remain; this is not a release speedup.

The [actual model capture](layouts-20260924.md) established 1,920 qualifying
attention layouts. The candidate changes only `TensorSlice.Reshape` and adds
one private helper. For exact dense row-major parent/slice types with unit
steps, it copies contiguous regions into independent dense storage. Nested,
stepped, reversed, empty, reduced and derived layouts retain the existing
fallback. It respects the parent's memory window and preserves gaps between
regions. Arithmetic and model graphs are unchanged.

Compiled review verifies **3,188 unchanged original Core methods**, original
implementation flags, equal public surface and the exact original fallback.
Only Reshape differs, with one new private helper. The root product remains the
selected release; the candidate is an isolated snapshot for qualification.

Each mode covers 343 existing tensor cases, 25 new copying cases and one
campaign-only identity case. The new contracts cover all nineteen observed
attention lengths, independent ownership, memory-window offsets, multi-axis
gaps, full slices, empty/reversed storage, stepped/negative/nested/reduced
slices, derived-type behavior and invalid target shapes. Float bit patterns
include signed zero, infinities and NaNs; copying must preserve every bit.

Two setup failures are retained:

- The first run passed 362 tests, including all copy cases and identity, but
  seven source-policy tests could not find the source root from the deployed
  runtime directory. Closure `84358a71` remains failed. The successor places
  the runtime under its source root and removes a forbidden null-forgiving
  operator from the helper, using an annotated nullable out result instead.
- The corrected candidate passed all 369 tests in normal mode. Its attempted
  disabled-mode run passed 368 but failed the hardware identity assertion:
  `DOTNET_EnableAVX512F=0` did not disable AVX-512. Closure `c7613d20` remains
  failed. [.NET 10.0.8 declares the group switch as EnableAVX512](https://github.com/dotnet/runtime/blob/v10.0.8/src/coreclr/jit/jitconfigvalues.h#L357).
  A separate successor used `DOTNET_EnableAVX512=0` with the same binaries and
  passed all 369 tests, including the hardware assertion. The passing normal
  run was reused; no product rebuild or repeat of that mode was needed.

All relevant owners are terminal. No performance clock is accepted from these
tests, and no earlier rejected optimization is reopened. The next gates are
actual-model numerical qualification, complete Slice→Reshape attribution and
the original controlled whole-application comparison. The optimistic observed
opportunity remains about 2.94 seconds per corpus; timing must prove the gain.

| Proof | SHA-256 |
|---|---|
| Candidate source receipt | `d11cfc00158da39984da41da3cafc4ceb404d2f9e002c542a701f244d30a5150` |
| Candidate Core | `bafdb0069c4809251fd4e287ad094076ed5e01cefe8be7be6dc6446f8887a9da` |
| Test consumer | `1d24ea12d4582f10745f2c3ba454e17072b203364bdba8a3f1592f3f221839a4` |
| Compiled scope review | `aa34820a580070972dde47225b1fa14a8172f6731ba91bb6b5aacfa30c527d96` |

The [complete qualification summary](qualification-20260924.json) binds both
TRX files, the final closure and retained failures. Raw evidence is under
`artifacts/parakeet-slice-materialization-build-amd-20260924`,
`artifacts/parakeet-slice-materialization-build-amd-v2-20260924` and
`artifacts/parakeet-slice-materialization-mode-amd-20260924`.
The corrected source is `artifacts/parakeet-slice-materialization-source-v2-20260924`.
Use [the correction controller](../slice-materialization-build-amd/correct_mode.py)
for provenance; its stage is closed and must not be observed again.
