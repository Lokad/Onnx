# Paradee bounded harmonic sine — 2026-11-06

Selected for integration, subject to main/committed-product qualification and a fresh actual native comparison.

Retain the isolated bounded harmonic sine:all24of24paired observations favor it,and every complete four-case two-sided95% source interval is below1. Short FP32 improves5.55%,mediumFP325.76%,shortINT85.06%,mediumINT83.07%. Complete dual-framework local/actual512numerical,path,private,scalar and native qualification passes with unchanged operator1e-4/wave1e-3/50dBgates. Fixed original-short parent paths are corroborated with zero event loss. Integrate only the two SHA-bound production/test files,then qualify the main and clean committed products and make a fresh actual ORTcomparison. These source gains do not establish final ORTparity.

Only the sine calculation inside the existing wide harmonic path changes.
For vectors entirely finite with absolute arguments at most 4096, the runtime
reduces the live, already-rounded arguments by a nearest integer multiple of
split pi, evaluates the independently derived degree-13 Taylor polynomial,
and restores parity and signed zero. Every other vector delegates unchanged
to Vector512.Sin. Frequency/phase multiplication, gain masking, per-harmonic
addition order, scalar/portable paths, allocation, output ownership, public
interfaces and dependencies remain unchanged. No new cache or public fast-math
setting is introduced.

## Complete fixed source comparison

All 56 workers and 24 pairs remain: eight conformance workers and six balanced
independent pairs per original FP32/INT8, 14/62-token workload. Each timing
process conditions for at least 32 calls **and 40 seconds**, then measures 33
complete Reset/Execute/waveform-copy calls. No observation was dropped, added
or replayed after inspection. Numerical work runs serially on CPU 2 of the
existing Standard_D4as_v7/AMD EPYC 9V45 VM, default .NET 11 RC3, with one common
GraphBench caller/dependency closure. Voice is 0; speeds are 1.0 and 0.8.

Times are arithmetic means of process means. Ratios are geometric paired means;
two-sided 95% log-Student-t intervals use df 5 and critical value 2.570581835636305.

| Original export | Tokens | Control ms | Candidate ms | Candidate/control | Two-sided 95% interval | Favorable pairs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP32 | 14 | 22.546 | 21.295 | 0.944521 | [0.936679, 0.952428] | 6/6 |
| FP32 | 62 | 60.379 | 56.913 | 0.942404 | [0.911720, 0.974121] | 6/6 |
| INT8 | 14 | 26.741 | 25.385 | 0.949424 | [0.919072, 0.980778] | 6/6 |
| INT8 | 62 | 61.798 | 59.905 | 0.969258 | [0.950356, 0.988536] | 6/6 |

These source results establish no Microsoft ORT ratio. The
[actual native checkpoint](paradee-runtime-20261106-b47.md) remains the current
comparison until replaced by a fresh committed-product native measurement.
Source ratios are not multiplied into earlier native results.

## Numerical and runtime qualification

The bounded polynomial has floating-point rounding differences; bitwise runtime
identity is not claimed. All original operator/upstream scaled-error bounds
remain at 1e-4, waveform peak at 1e-3 and signal-to-error at least 50 dB. Direct
sine tests require absolute error at most 3e-7 against independent double sine;
synthetic fused-output tests use the tighter 1e-5 scaled bound. Unchanged modes,
inputs/held outputs and large/special-vector fallback preserve their bit tests.

Untimed arithmetic feasibility checks 21,727,394 arguments per framework,
including SHA-verified real operands, domain samples, signed zeros, special
values and pi neighbors. Maximum absolute error against independent double
sine and BCLVector512.Sin is 1.1920929e-7. Complete captured harmonic operator
checks have maximum scaled error 6.67572021484375e-6. A separate software-wide
full-wave diagnostic is retained as distinct diagnostic evidence, never as
production hardware or timing proof.

Both exact local products pass 6,249 ordinary tests with the same 73 explicit
skips, 308 default tests with five AVX-512 skips, 282 focused scalar tests with
25 hardware skips, all six FULL hardware-disabled model facts over eight inputs
twice, and 343 native differential cases. Both actual Linux products pass 313
default facts without skips, including the direct sine/fallback tests and all
private models, 282 focused scalar facts with 25 explicit skips, 48 path
conditions, 14 harmonic self-checks, 16 captured operands and 28 independent
native Fourier regions. Existing BCL512-versus-portable harmonic controls stay
bitwise equal; new runtime-versus-control error is explicitly measured at the
unchanged 1e-4 operator bound. Duration, mutation/reset/repeat/random-state and
held-output requirements remain intact.

The fixed late trace inventory records harmonic parent-path presence with zero
lost events. Sampled frames establish neither dimensions nor unprofiled CPU
shares, cycle savings or the identity of every inlined helper.

## Identities and evidence

Source base is `97dad5121da71b4e8fa0bcf38901403c6a56d17a`. Only
CPUExecutionProvider.HarmonicSum.cs and HarmonicSumFusionTests.cs change. The
exact measured clean b47 baseline Core
`f045990b351d4685bc5bb08810e3ca415be84be2a3326385d22fb7188a15b5b5` retains its original compiler
identity; all 143 canonical baseline runtime/global bodies match the report-only
base. Timed candidate Core is `c3fd514f75098b1dc3643ce4c442434e3e23959a2624fc9f72ab65db6f5baf35`; common caller is
`0c629efcc31a6c19c44be78e538a29f4bbe9a4810b3881aa396692349f6212db`.

The [complete JSON](paradee-harmonic-sine-20261106.json) retains every worker
clock, supervision record, pair, qualification, source/product identity, trace
inventory and decision. Full 169,986,401-byte archive SHA256 is
`ec104d7b6916895b4e8bf8d79acf29d34754291b9e44927f274e288bb6dfd6f7`. Actual slot 053 was released at
`2026-11-06T23:19:48.380218+00:00` after full local SHA/size verification,
all 18 original model/input/reference hash checks, a global empty-worker audit
and matching-token removal. Only the fully exported own root was deleted.

Final all-four actual ORT parity, exact committed-product qualification,
established-model controls, the under-60-minute seven-model routine and final
terminal-green CI remain mandatory. No package, release or tag is produced by
this source experiment.
