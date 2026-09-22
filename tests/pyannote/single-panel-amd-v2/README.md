# Corrected single-panel AMD preparation

The first preparation stopped before deployment because its selftest imported
an omitted helper, `prepare_execution.selected_native_files`. All 28 other tests
passed. That failure remains closed at
`0f7a304a1fb33ca7c96cee50a9045d5faa53d0c89599bf57542b54f3f0f0f042`.
This successor restores the exact required helper in a new tool/output directory.
Candidate bytes, native assertions, timing schedule and every gate are unchanged.

## Prospective AMD application campaign

Compare accepted production Core `e9c87932` / Data `85d166b5`, the newly
qualified single-panel candidate Core `1279b4b6` / Data `4e602d9f`, and fresh
Microsoft ONNX Runtime 1.29.0. Pyannote remains first, Parakeet second, Whisper
deferred. This candidate changes actual packing ownership: admitted patches
of at most 32 columns are read directly, with no packed-array rental or copy.
All numerical kernel bodies remain identical to the qualified component.

The preceding direct-output application trial stays unselected. Its complete
candidate/production ratio 0.970383 missed the fixed 0.970000 limit, despite
passing all correctness and repeatability checks. This new product is a distinct
candidate; no old sample is reused for performance selection and no gate changes.

Before timing, build normal candidate source on Linux and compare all 3,113
Core / 697 Data methods and public declarations with the frozen Windows runtime.
Run full backend/tensor suites, requiring at least 3,313/343 passes and the
original mandatory hardware test. Execute all 400 actual-caller cases in both
normal and hardware-disabled modes. Preserve every non-NaN bit and NaN
classification; retain all differing payloads as diagnostics.

Run fresh Pyannote and Parakeet qualification for both managed roles: 36
Pyannote arrays / 32 public requests and 1,568 Parakeet arrays. Candidate Pyannote
arrays and public results must match production exactly, and every original
native bound and application check must pass. The separate Parakeet arithmetic
candidate and larger packing budgets are not included. All Windows graph,
shared-model, caller, suite and independent package evidence is pinned.

Fresh native conformance and both 600-second meetings plus 30-second recovery
precede timing. Require unchanged native speaker timelines, centroid bounds,
input ownership and held results. Then execute six fresh processes in order:
production, candidate, ORT, ORT, candidate, production. The existing protocol
calls the candidate role `portable`. Each process has one warmup and three
measured passes over the full 30-second dialogue and its three ten-second crops:
96 total requests, 24 warmups and 72 measurements. Retain every sample.

All roles must have process-mean max/min <=1.10 on the full dialogue and <=1.20
on each crop. Candidate / production must be <=0.97 full and <=1.05 per crop.
All twelve controls and all four speed gates are mandatory. Failed controls or
speed gates stay failed, with no unchanged timing retry. The complete application
Lokad/ORT <=1.05 target remains separate and is not replaced by a component gain.

CPU2 affinity precedes runtime startup; monitor CPU0. .NET 10.0.8, SDK10.0.204;
ORT uses one intra/inter-op thread, sequential execution, full optimization and
no spinning. No profiling/runtime overrides during timing. Limits remain four
hours per campaign, one hour per worker, 12 GiB owned RSS, 1 GiB available
memory/tmpfs and 2 GiB artifacts. Preflight requires 12 GiB memory / 3 GiB tmpfs.
The terminal-aware monitor from the preceding campaign is unchanged.

From repository root, run `C:/Python313/python.exe -X utf8 -B` with `prepare.py`,
inspect successful selftests, then `finish.py` once. Artifacts:
`artifacts/pyannote-single-panel-amd-payload-v2-20260922` and
`artifacts/pyannote-single-panel-amd-execution-v2-20260922`. Remote directory:
`/dev/shm/lokad-pyannote-single-panel-app-v2-20260922`. The exclusive VM authorization
is already in force. Collection waits for terminal owners; an independent audit
checks every qualification, raw timing and resource gate before publication.
Root integration requires a passing complete verdict and normal root/package checks.
