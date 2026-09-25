# Direct Parakeet depthwise candidate: focused numerical qualification

**All eight tests pass in both normal and hardware-disabled processes.**
Each process checks **57,332,736 output values across all 59 actual
geometries** against the separately loaded original M78 Core. Finite
outputs and signed zeros agree bit-for-bit; explicit special-value tests
also preserve NaN classification. No performance result follows from this.

The candidate directly accumulates nine taps in the existing dense layout.
It retains the original FMA positions, multiply/add tail and sum-then-bias
order. Explicit segmented selection and unsupported geometry, batch, degree
or hardware options keep their existing paths. Output allocation is unchanged.

The tests cover full observed geometries, spatial borders and flattened
vector tails, line tails and memory offsets, special values including
padding, fallback options/geometries, logical views and independent outputs,
the 1D provider, dirty pooled spatial outputs, and runtime/product identities.
Eligible normal-hardware operator calls report zero scratch; complete
application mechanism counters remain a separate requirement.

| Compiled scope | Verified |
| --- | ---: |
| Changed original Core methods | 1 |
| Added private helpers | 4 |
| Other original Core methods unchanged after exact metadata reconciliation | 3,276 |
| Data methods unchanged | 697 |
| Added warnings | 0 |

Data is reused byte-for-byte. Three compiler-generated private names shifted
by four and two callers refer to those shifted names; every instruction,
local, exception region and implementation flag otherwise matches exactly.
The initial reviewer refusal and the additive reconciliation are retained.
Three rejection tests cover instruction, flag and unexpected-name changes.

The first build compiled Core but failed in the new test harness: generic
type inference for TryGetArray and a platform-analysis warning. The corrected
source changes only those two test expressions. Every product source byte
matches the first candidate. The failed build remains recorded.

Candidate Core SHA256: `40260aef7fd93c5153601ec104a87843a2c017a2720fd64c9e24e3460d455749`.
Data SHA256: `01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1`.
Closure SHA256: `26e4da0a66bf37aeda09dd5b0a5144817585ab7311cf87f85c4bb9ded894d171`.

All 242 resource observations pass. Peak owned RSS is 1,699,147,776 bytes.
Both workers and their owner are terminal with exit code zero. The full
transcription mechanism capture, native-model correctness, shape-weighted
screen and complete application/regression admission remain outstanding.
M78’s independent e5 failure still prevents release promotion.

[Per-geometry evidence and test names](contracts-20260925.json),
[causal diagnosis](../depthwise-route-results/diagnosis-20260925.md).
