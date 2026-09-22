# Current Parakeet versus Microsoft ONNX Runtime on AMD

Refresh the complete twenty-clip baseline for integrated M22 Core `208371f6`
/ Data `b9358370` against ORT 1.29.0. Use unchanged qualified AudioBenchmark
and native consumers, original models/audio/references and the complete native
dependency inventory. The exact M22 candidate manifest is copied byte for byte;
its role is now the current selected product. Complete product, model, native,
meeting, root and package qualification is retained and checked, not rerun.

Four fresh processes run current, ORT, ORT, current. Each runs every clip once
as warmup and three times measured: 320 requests, 80 warmups and 240 measurements.
Every raw clock and setup interval is retained. The twenty clips contain
213.265 seconds of audio. Means use exact integer-clock fractions, three passes
per process and two equally weighted processes per role. Corpus time sums all
twenty clip means. No sample from an earlier campaign enters this result.

Require all 42 process-repeatability controls: max/min <=1.10 for the corpus
and <=1.20 for each clip, for both roles. This is a baseline refresh with no
product change, so no candidate-versus-current improvement gate applies.
The independent parity target remains current/ORT <=1.05. An unstable result
is retained as invalid without sample removal or an unchanged retry.

Validate every complete transcript/token/duration result, readonly input and
held-output contract. All current results must equal the closed M22 public
reference. Preserve original native tolerances/settings and loaded-library
checks. Request timing includes frontend, neural graphs, greedy decoding and
owned results, with model loading, file IO and external validation separate.
All worker threads inherit CPU2 before runtime startup; the monitor uses CPU0.
ORT runs sequentially with one intra/inter-op thread, full optimization and
disabled spinning. No numerical overrides or profiler are enabled.

Bounds: 12 GiB available memory and 3 GiB tmpfs before workers, 12 GiB owned
RSS, 3,600 seconds per worker and four hours overall, 1 GiB minimum available
memory/tmpfs and maximum worker output, 2 GiB total artifacts. Preserve process
snapshots/foreign-CPU accounting and its short-lived-process limitation.

From repository root run `C:/Python313/python.exe -X utf8 -B` with this
directory's `selftest.py`, then `run.py prepare`, `run.py stage`, `run.py launch`.
Observe with `run.py observe`. After all recorded owners are terminal, run
`run.py collect` and `audit.py`. Destinations must not exist beforehand:
`artifacts/parakeet-current-baseline-amd-20260922` and
`/dev/shm/lokad-parakeet-current-baseline-20260922`.

Five checker tests cover the retained real records, every mandatory control,
incomplete/duplicated requests, ownership/transcript/native-setting changes,
all prior qualifications, and exact reconstruction of the previously published
Core `1279b4b6` baseline from its original raw clocks. Preparation and auditing
do no model execution or product build on the workstation.
