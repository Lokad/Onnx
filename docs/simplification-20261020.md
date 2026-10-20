# Runtime and maintenance simplification

This work starts from `f11777fb91eeff930db8d5a545ffe3481e9f517f` and retains the
qualified normal runtime paths and complete public API. The implementation is
`46f44e04d6c9440f8b4311391f472ae9140f07ec`.

Closed experiment producers and retired code are indexed in
[the Git-history archive](../archives/README.md). All 5,995 original paths were
extracted and compared to their original Git blob hashes. All forty formerly
hash-locked C# sources also pass their original normalized SHA-256 checks.
Historical reports/results and ignored models/raw evidence remain available.
Active test Python shrinks from 5,478 files / 436,079 lines to 126 files /
10,351 lines. The path index is generated archive metadata, rather than a new
source-policy gate.

The runtime retires fourteen default-off environment controls and their
exclusive internal implementations. Nine qualified comparison controls remain.
Public methods and options stay compatible, including segmented convolution and
Memory mode. The handwritten optional-parameter scanner, TODO gate and
SDK-internal Roslyn tool are archived. Maintained guidance is in
[code maintenance](code-maintenance.md) and
[runtime options](runtime-options.md).

The [routine benchmark command](../eng/bench.md) reuses the existing validated
workers, alternates cores/engines and retains every call. It has no mandatory
A/A calibration or cooldown. Strict protocols remain available for historical
qualification and release claims. CI now classifies changes: prose skips the
numerical matrix, while packaged prose also gets package smoke. Unknown code
changes conservatively run the complete existing matrix.

## Correctness and compatibility

| Check | Result |
|---|---|
| Complete compiled public API | All 2,830 declarations unchanged; no additions, removals or metadata changes |
| Common method implementation flags | Unchanged |
| Package contents and independent consumer | Pass |
| Native differential contracts | 309 pass |
| e5 tokenization, hidden states and embeddings | Nine contracts pass across scalar/SIMD/intrinsics |
| Ordinary tensor suite | 414 pass, 10 hardware skips |
| Ordinary backend suite | 3,597 pass, 23 skips; one unchanged Windows ResNet hash failure |
| Scalar tensor/backend suites | 406 / 3,400 pass, with 18 / 221 hardware skips |
| Python protocol and maintenance fixtures | 196 pass |
| Retained audio replay/example projects | All ten build |
| Complete Gemma outputs | 75 arrays per mode match the control byte-for-byte in scalar, SIMD and default; complete native-reference checks pass |

The Windows ResNet failure exactly repeats the preceding qualification:
actual hash `8566038335043033094`, expected `15780008002253672869`. No hash or
numerical expectation was changed to obtain a pass.

LSTM projection tests now require exact bits for every finite result, signed
zero and infinity, and require NaN classification for exceptional results.
Both original reductions remain unchanged. The old intermittent NaN-payload
mismatch did not reproduce in the isolated Windows/Linux scalar probes. A
separate operand-order probe produces `7fc12345` and `ffc54321` for the two
orders of the same NaN multiplication: payload selection is representation
sensitive, and is not a portable arithmetic contract. This establishes the
reason for the narrower assertion without proving the original CI scheduling
cause. Negative fixtures reject changed finite bits, zero/infinity signs and
NaN/finite mismatches. The complete projection tests pass on Linux scalar
without retries; guards, immutability and allocation assertions stay enforced.

The initial CI run passed eight jobs and failed the Ubuntu vector allocation
check in `PackedColumnRemainderTests`: it observed 1,136 bytes instead of zero.
The measurement interval now lives in a separate non-inlined, optimized test
helper, keeping test-loop tier promotion and assertions outside its counters.
The production kernel and zero-allocation requirement remain unchanged. Local
focused checks pass; no failed CI job was retried.

Removing legacy-routing hooks left one reference softmax with only test callers.
That method now lives in shared test support, with unchanged arithmetic. All ten
affected softmax/LSTM contracts pass normally and without hardware intrinsics.
The remaining runtime IL matches the measured core after accounting for the
compiler-generated member-number shift caused by that relocation.

## Practical timings

These are two balanced process pairs on the existing AMD VM, comparing
f11777fb with implementation 46f44e04. Each engine has two fresh processes per
case and 33 retained calls per process. Gemma includes the owned 768-dimensional
vector; tokenization/model loading are outside its timer.

| Case | Candidate | ORT | Paired candidate/control | Candidate process spread |
|---|---:|---:|---:|---:|
| Gemma FP32, 128 tokens | 165.51 ms | 167.58 ms | 0.9983 | 0.6% |
| Gemma FP32, 512 tokens | 785.90 ms | 739.08 ms | 1.0013 | 0.1% |

Both observations are consistent with no material change. They are descriptive
engineering comparisons, with no calibrated sub-percent improvement claim.
The later follow-up changes reference-code placement and benchmark/test guards;
the measured model arithmetic stays the same. Published benchmark headline
figures remain unchanged.

## Retained evidence

Build, archive recovery, numerical, package/API and practical timing receipts
are local under ignored `artifacts/simplification-20261020`. The original
cleanup remains independently documented in [its report](cleanup-20261019.md).
