# Untimed Where fallback code generation

The single fixed M56 screen is rejected at fd847767:52repeatability failures
and40per-case regressions. Selected source81f75c38 remains the release. This
diagnostic does not retry performance admission, change its score, trim clocks,
or support a new timing claim. Both product binaries remain unchanged.

Run two fresh ordinary-tier processes, selected then V3, with only
DOTNET_JitDisasm="Lokad.Onnx.Tensor`1:Where Lokad.Onnx.UniformScalarWhere:Try Program:Exercise".
Capture emitted Where, new helper and consumer bodies. Use the same122valid
qualified cases, in the exact fixed-screen order, with the same deterministic
batch sizes and120repetitions:8,820,240public calls/process. No Stopwatch,
frequency, timestamp or clock collection exists in the new execution loop.
The qualified fixture construction and coordinate oracle are byte-equivalent
apart from the documented partial-class/entry-name changes. All output bits,
full stores/guards, held outputs and independent ownership checks remain.

Each case emits prepared and complete journal entries. Retain both full stdout
streams; review every emitted body, tier, call target, loop and code size, with
resolved branch labels. Compare whether float fallback reaches an optimized
whole-method entry or only an optimized loop entered from instrumented code.
Examine the consumer's direct call sites as well. Do not infer the historical
timed workers' tiers from this separate execution. Ordinary tiering is untouched;
no TieredCompilation/QuickJit/PGO override or product method flag is changed.

The probe is bounded to five serial jobs:SDK,restore,build,current,candidate.
CPU2compute/CPU0monitor, .NET10.0.8/SDK10.0.204, no Windows build or inference.
12GiBavailable/3GiBtmpfs preflight,8GiBRSS,1GiBremainingmemory/tmpfs,900s/job,
fourhours/campaign,1GiBoutput/job,2GiBtotal artifacts. All PID/birth owners and
immutable dependencies must verify before starting. Preserve the two closed
numerical and performance campaign receipts and files.

Use C:/Python313/python.exe -X utf8 -B with
tests/parakeet/scalar-where-fallback-codegen/run.py prepare,stage,launch,observe,
collect, then its audit.py. Refuse existing namespaces; freeze before preparing.
Local artifacts:parakeet-scalar-where-fallback-codegen-amd-20260924.
VM:/dev/shm/lokad-parakeet-scalar-where-fallback-codegen-20260924.
Numerical inputs are linked after digest verification; no new models or reference
downloads. All original performance gates remain frozen. A follow-up source
candidate requires a concrete new mechanism supported by this review and full
build/numerical/codegen/screen qualification before any application or integration.
