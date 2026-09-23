# Current Parakeet complete-request profile on AMD

Prospective diagnostic: unchanged Core521bae17/Dataf3b9aa81, complete20clip
213.265second corpus, control/sample-a/sample-b, one warmup plus three measured
passes each. All240public calls and clocks must pass the original full native
reference and exact current public results, input and owned-output checks.
Build closure27df0af0 qualifies the narrow SampledAudio consumer adaptation.
One explicit FullParakeet stack marker covers the complete60measured calls in
each capture; its total must agree within5% of summed wall time, on the target
native thread. Keep every profile, warmup exclusion, both independently reconciled
Speedscope/Chromium exports, all leaf/inclusive weights and control overhead.
The inherited EventSource name Lokad-Pyannote-Diagnostic is transport only.

New full-Parakeet capture bounds are fixed before execution:12GiB available
before each target,12GiB combined owned RSS,900seconds per pair,1GiB minimum
available/tmpfs,1GiB output per pair,2GiB total artifacts. Initial tmpfs3GiB.
The qualified baseline peaked9.50GB; old Pyannote8GiB cap is unchanged in its
closed experiment. Targets CPU2; collector/monitor CPU0. Exporters retain8GiB
RSS/available,900seconds,3GiB initial tmpfs and1GiB minimum live headroom.

Use C:/Python313/python.exe -X utf8 -B prepare.py, then transport.py stage,
launch,observe,collect; export.py launch,observe,collect; audit.py. All actions
are separate commands and refuse existing output. No product tuning, downloads,
timer trimming, favorable retry or new ORT ratio. Select an optimization only
after reading current AMD attribution and relevant Lokad/Microsoft ORT source.
