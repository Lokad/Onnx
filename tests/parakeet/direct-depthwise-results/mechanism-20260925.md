# Direct depthwise: the predicted overhead disappears in the complete application

All **80 public transcription requests** match the original M78 results
exactly: transcripts, tokens and decoder calls. Every targeted convolution
takes the direct route across **all 59 observed geometries**. All 2,080
operator calls finish through that route over four corpus passes.

| Work per 20-clip corpus | Original M78 | Direct candidate |
| --- | ---: | ---: |
| Operator calls | 520 | 520 |
| Tiled panels | 8,872 | 0 |
| Generic matrix products | 3,948,544 | 0 |
| Temporary group tensor views | 11,845,632 | 0 |
| Float values written into patches | 1,084,870,656 | 0 |

All 520 eligible calls per corpus use direct accumulation. Source/dense
layouts and frequencies match the original observation at every geometry.
No target call rents tiled scratch or enters a generic matrix leaf.

The observer changes only three callsites; all four optimization helpers
remain unchanged. It adds 16 private observer methods, with 3,278 original
Core methods and all 697 Data methods unchanged. Data and the common
public consumer are reused byte-for-byte; there are no added warnings.

All 434 resource samples and original CPU-accounting checks pass. Peak
owned RSS is 8,135,192,576 bytes. The actual owner and worker are terminal
with exit code zero. These instrumented times are deliberately unscored.

This confirms the proposed mechanism. Full model arrays must still pass
the native ORT comparison, followed by the shape-weighted screen and
complete application/regression gates on the uninstrumented candidate.
The root product and BENCHMARK.md retain their qualified release results.

Uninstrumented Core: `40260aef7fd93c5153601ec104a87843a2c017a2720fd64c9e24e3460d455749`.
Mechanism closure: `ac82181334d170d9a1f1d36eca08f4e7bc85b303effd57cf8a0d225455119a8c`.

[All per-geometry counts](mechanism-20260925.json),
[focused numerical qualification](contracts-20260925.md),
[original ORT-guided diagnosis](../depthwise-route-results/diagnosis-20260925.md).
