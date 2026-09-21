# Prospective pooled-convolution application comparison

This experiment compares pooled Core `0d098ba5` with predecessor Core
`469cb2d6` and Microsoft ORT 1.29.0. Both managed roles use request-context
Data `1d346664` and the original public AudioBenchmark consumer. It changes
the convolution allocation implementation; it does not retry the earlier
Data-only experiment unchanged.

Complete graph, shared-model, affected Parakeet, full-suite and public
dialogue/meeting qualification must pass before preparation. Run with
`C:/Python313/python.exe -X utf8 -B`: first `prepare.py <application-closure-sha256>`,
then `run.py`, then `audit.py` after all recorded processes are terminal.
Preparation pins every runtime, tool, model and input and refuses existing
output. The artifact directory is
`artifacts/pyannote-convolution-pool-comparison-20260921`.

The frozen order is predecessor, candidate, ORT, ORT, candidate, predecessor.
Each process has four warmup and twelve measured requests, for 96 calls total.
All complete application/native/ownership checks remain in force. CPU2 and
normal .NET 10.0.12 are retained; no forced GC or altered runtime settings.
Limits are 10 GiB available before launch, 8 GiB RSS, 1 GiB minimum available,
20 GiB free disk, 1 GiB output and 1,800 seconds per worker.

All roles must pass process-mean max/min limits of 1.10 for the complete
30-second request and 1.20 for each fixture. Admission additionally requires
at least 3% lower full-request latency with no fixture regression above 5%.
Keep every observation and failed control. A failed experiment does not
authorize an unchanged retry. These Windows measurements do not establish
calibrated parity or modify the frozen primary AMD qualification payload.
