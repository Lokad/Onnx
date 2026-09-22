# M22 complete Pyannote application comparison on AMD

Compare the exact measured selected Core `3c2f16b0` / Data `6318cf48`, LSTM
candidate Core `208371f6` / Data `b9358370`, and Microsoft ONNX Runtime 1.29.0.
Reuse the closed normal Linux suites/package, Pyannote, Parakeet and shared/e5
qualification for those exact products. Verify each closure and its complete
analysis before workers. No completed suite or tensor campaign is repeated.

First run fresh native public conformance on four Pyannote fixtures and twenty
Parakeet clips. Require all original public contracts, pinned native libraries,
one intra/inter-op thread, sequential execution, full optimization and no
spinning. Run both 600-second meetings and the 30-second recovery request with
the candidate, preserving complete speaker timelines, centroid tolerances,
readonly inputs and retained outputs. Consumers and native scripts are unchanged.

Then run six fresh processes: selected, candidate, ORT, ORT, candidate, selected.
Each performs one warmup and three measured passes over the full 30-second
dialogue and three ten-second crops: 96 requests, 24 warmups and 72 measurements.
Keep every clock and every preparation interval. Candidate complete public
results must equal fresh selected execution, including across processes.

Preserve all twelve repeatability controls: process-mean max/min <=1.10 for
the full dialogue and <=1.20 for each crop, for every role. Candidate/selected
must be <=0.97 for the full dialogue and <=1.05 for every crop. All controls and
four speed gates are mandatory. Application parity remains the separate
Lokad/ORT <=1.05 target. An unchanged rejected timing trial is not rerun.

The timer covers the complete public request: frontend, inference and owned
results. Model setup is recorded separately. Preserve process snapshots and
foreign CPU accounting, with the existing limitation that snapshots can miss
short-lived processes. CPU2 affinity precedes CLR/native startup; monitoring
uses CPU0. No LOKAD/DOTNET/COMPlus overrides are allowed.

Freeze 12 GiB available / 3 GiB tmpfs preflight, 12 GiB owned RSS, 3,600 seconds
per worker and four hours per campaign, 1 GiB minimum available memory/tmpfs
and output, and 2 GiB total artifacts. Native Python source and binary inventory
includes the entire preceding execution inventory, in addition to model and
SDK dependencies. Local preparation/auditing performs no build or inference.

From repository root use `C:/Python313/python.exe -X utf8 -B` with this
directory's `selftest.py`, `run.py prepare`, `run.py stage`, `run.py launch`.
Use `run.py observe`; after terminal state run `run.py collect`, then `audit.py`.
New namespace: `artifacts/pyannote-lstm-input-app-amd-20260922` and
`/dev/shm/lokad-pyannote-lstm-input-app-20260922`. Preserve failures and all raw
evidence. Root integration still requires admission and normal root/package
verification; no previous performance sample enters the new verdict.
