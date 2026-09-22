# Prepared convolution: prospective complete Pyannote comparison

Compare selected production Core `1279b4b6` / Data `4e602d9f`, the prepared
convolution candidate Core `3c2f16b0` / Data `6318cf48`, and fresh Microsoft ONNX
Runtime 1.29.0. Pyannote remains first, Parakeet second, Whisper deferred.
No previous timing sample contributes to this campaign's verdict.

The exact candidate has passed complete Windows suites, package consumption,
Pyannote/public models, shared/e5 regressions, and selected-versus-candidate
Parakeet regression. The latter preserves the three known Windows native
mismatches and all twenty public clips; it is not a native numeric pass.
Actual product AMD AVX2/AVX512 raw and captured-layer qualification is closed at
`88ec9b71d6e7807440ccd65d6148ead8c585a0633ce755898e174b8501b77197`.
Reuse this exact-runtime evidence: 8,004 raw graph requests and 108 captured
layer cases / 216 calls in each width. Check its closure, summary and candidate
identity before full-model execution. The old 400 single-panel caller cases
belonged to the preceding optimization; this candidate's own complete ordinary
graph coverage replaces them. All other campaign gates are retained.

Before timing, build the normal candidate source on Linux, compare all 3,161
Core and 697 Data method bodies and public declarations, and run full backend
and tensor suites with at least 3,344 / 343 passes and the mandatory hardware
test. Fresh qualification for both managed roles covers 36 Pyannote arrays,
32 public diarization requests, and 1,568 Parakeet native arrays. Both AMD native
numeric gates must pass. Candidate Pyannote arrays and public outputs must match
selected production exactly. Run fresh native public conformance and both
600-second meetings plus the 30-second recovery request, preserving speaker
timelines, centroid bounds, inputs, and held outputs.

Run six fresh processes in order: production, candidate, ORT, ORT, candidate,
production. The inherited protocol calls the candidate role `portable`.
Each process has one warmup and three measured passes over the full 30-second
dialogue and three ten-second crops: 96 requests, 24 warmups, 72 measurements.
Retain every sample. All three roles must have process-mean max/min <=1.10 on
the full dialogue and <=1.20 on every crop. Candidate/production must be <=0.97
on the full dialogue and <=1.05 per crop. All twelve repeatability controls and
four speed gates are mandatory. An unchanged failed performance trial is not
retried. Application parity remains the separate Lokad/ORT <=1.05 target.

CPU2 affinity precedes CLR/native startup; monitoring uses CPU0. The target is
.NET10.0.8 / SDK10.0.204. ORT uses one intra/inter-op thread, sequential execution,
full optimization, and no spinning. No profiling or runtime overrides during
timing. Existing limits remain four hours per campaign, one hour per worker,
12 GiB owned RSS, 1 GiB minimum available memory/tmpfs, and 2 GiB artifacts.
Preflight requires 12 GiB available memory and 3 GiB free tmpfs.

From the repository root run `C:/Python313/python.exe -X utf8 -B` with this
directory's `prepare.py`, then `finish.py` once. It stages, launches, observes,
collects only terminal processes, and independently audits results. Artifacts
are `artifacts/pyannote-blocked-spatial-app-amd-payload-20260922` and
`artifacts/pyannote-blocked-spatial-app-amd-execution-20260922`; the remote path
is `/dev/shm/lokad-pyannote-blocked-spatial-app-20260922`.
Root integration requires admission and normal root/package verification.
