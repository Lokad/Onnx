# Paradee direct wide convolution output — 2026-11-07

Selected for integration, subject to main/committed-product qualification and a fresh actual native comparison.

Retain the bounded direct wide convolution output path:short FP32 improves3.90%,ratio0.960958 with two-sided95% interval[0.936374,0.986186],all6pairs favorable. MediumFP32,shortINT8 and mediumINT8 estimates improve2.69%,3.71% and0.94%,but their complete intervals include no change;those gains remain unproved. Allfour source upper endpoints are below1.03,with18of24favorable pairs and no unresolved material source regression. Both exact local/actual512products pass unchanged numerical/private/scalar/native gates,including all3newmode hardware facts;fixed packed parents occur in both zero-loss traces. Integrate only the four SHA-bound source/test files,qualify main and clean committed products,and make a fresh original-export ORTcomparison. Source estimates do not establish native parity or final acceptance.

The candidate reuses the existing packed AVX-512 12/8-row matrix kernels for
eligible float convolutions, seeding accumulators with positive zero and writing
directly into the final strided output. Full vector lanes preserve ascending
FMA order; final masked lanes keep separate multiply/add order. Bias is added
once after reduction. General matrix products retain their accumulating mode,
and existing convolution fallbacks remain. No new cache, dependency or public
setting is introduced.

## Complete fixed source comparison

All 56 workers and 24 pairs remain: eight conformance workers and six balanced
independent pairs for each original FP32/INT8, 14/62-token workload. Each timing
process conditions for at least 32 calls **and 40 seconds**, then measures 33
complete Reset/Execute/waveform-copy calls. No observation was dropped, added
or replayed after inspection. Numerical workers run serially on CPU 2 of the
existing Standard_D4as_v7/AMD EPYC 9V45 VM, using default .NET 11 RC3 and one
common GraphBench caller/dependency closure. Voice is 0; speeds are 1.0 and 0.8.

Times are arithmetic means of process means. Ratios are geometric paired means;
two-sided 95% log-Student-t intervals use df 5 and critical value 2.570581835636305.

| Original export | Tokens | Control ms | Candidate ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 21.334 | 20.499 | 0.960958 | [0.936374, 0.986186] | 6/6 |
| FP32 | 62 | 57.062 | 55.546 | 0.973111 | [0.937722, 1.009836] | 4/6 |
| INT8 | 14 | 25.072 | 24.147 | 0.962908 | [0.908731, 1.020315] | 5/6 |
| INT8 | 62 | 59.660 | 59.100 | 0.990618 | [0.963831, 1.018149] | 3/6 |

These source results establish no Microsoft ORT ratio. The
[actual native checkpoint](paradee-runtime-20261107-bb2.md) remains the current
comparison. Source ratios are not multiplied into earlier native results.

## Qualification and recovery

Both exact local products pass 6,249 ordinary tests with the same 76 explicit
skips, 338 default facts with ten hardware skips, 312 focused scalar facts with
30 explicit hardware skips, all six FULL hardware-disabled model facts over
eight inputs twice, and 343 native differential cases. Both actual Linux
products pass 347 of 348 default facts: only the hardware-absence refusal test
skips. All three new wide-output facts execute, covering independent arithmetic
oracles, strided and accumulating modes, live edits, special values, output
guards, storage refusal and untouched unused temporary storage. Both products
also pass 312 focused scalar facts, 48 model-path conditions, 14 harmonic
self-checks, 16 captured operands and 28 independent native Fourier regions.

All original numerical limits remain: operator/upstream scaled error at most
1e-4, waveform peak at most 1e-3 and signal-to-error at least 50 dB. Duration,
mutation/reset/repeat/random-state and held-output checks remain intact.

The previous isolated 085 attempt failed the actual AVX-512 unused-temporary
test before timing. Candidate 086 moves the old remainder clear below the new
early return; that failure and its independent diagnosis remain retained.
Preparation initially stopped on an old scratch-equality expectation. An
independent fixture-geometry calculation proves exactly which redundant packed
panel rents disappear; no request is added and all other shape/cache/copy/path
fields remain equal. Recovery reused completed operations and finished only
missing checks, with no source/product or numerical-bound change.

The two fixed late traces corroborate the declared PackedTile12 parent with
zero lost events. They establish no direct zero-seed/stride-mode, dimension,
unprofiled CPU-share or cycle-saving claim. The existing local trace auditor
needed only an 086 argument-allowlist entry; its original copy and exact change
proof are retained. No trace gate or declared frame changed.

## Identities and evidence

Source base is `f0e5114568afeb09a095e049735056cea31c086d`. Exactly four files change:
MathOps.PackedAvx512.cs, MathOps.PackedColumnTails.cs, Zzz.ConvDirectOutput.cs
and ConvDirectOutputTests.cs. The exact measured clean bb2 baseline Core
`6e2f4e59226d36f7450ccc842a6ee3d6f81fd33ffbb79c7d7024335c04c6c37a` retains its original compiler
identity. All 143 canonical baseline bodies match the report-only source base.
Timed candidate Core is `df4aca23237aeff451ac9f5f21fcdf7fd7f1883cb99eeaf45db4b14827d93353`; common caller is
`773accae5af1c1946c4baed9056bcd99ea527b25c3cdc0e62ec0341b8725405a`.

The [complete JSON](paradee-direct-wide-conv-20261107.json) retains every worker
clock, supervision record, pair, qualification, source/product identity, trace
inventory and decision. The full 170,209,123-byte archive SHA256
is `cd939ae7ac86d41fa714f9b937df2da796516050aacf5f0f34d4cee6db6adecd`. Preparation under slot 056 was exported and
released before exact-product reuse under fresh slot 057; qualification was
not replayed. Slot 057 was released at `2026-11-07T01:56:55.151032+00:00`
after complete local SHA/size verification, all 18 protected asset hashes,
a global empty-worker audit and matching-token removal. Only the fully
exported own root was deleted.

Final all-four actual ORT parity, exact committed-product qualification,
established-model controls, the under-60-minute seven-model routine and final
terminal-green CI remain mandatory. This source experiment produces no release
or tag.
