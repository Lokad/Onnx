# Parakeet regression for prepared convolution

The Pyannote candidate changes shared convolution preparation and execution.
Run both the currently selected runtime (`1279b4b6` / Data `4e602d9f`) and the
unselected prepared-convolution candidate (`3c2f16b0` / Data `6318cf48`) through
the unchanged TranscribeReplay and AudioBenchmark consumers. No arithmetic
candidate is used as the selected control.

From the repository root use `C:/Python313/python.exe -X utf8 -B` with this
directory's `run.py prepare`, `run.py execute`, and terminal-only `run.py audit`.
The destination must be new. Four sequential CPU2 workers cover all 784 native
arrays / 3,090,494 values and all twenty public clips in each runtime. Every
native array and public output must remain identical to selected production.

The selected Windows runtime has three known native failures at decoder step26:
english-16k, english-frame-limit, and english-repeat (`outputs`). Preserve exactly
those failures and their numerical maxima. A successful regression does not
claim the native numeric gate passes. The original independent native and public
auditors recompute their checks from all retained outputs.

Native preflight requires 12 GiB available and limits RSS to 8 GiB; public
preflight requires 14 GiB and limits RSS to 12 GiB. Each worker is limited to
1,800 seconds, 1 GiB output, 1 GiB minimum available RAM, and 20 GiB free disk.
All PID/birth identities, resource observations, inputs and tool digests are
retained. This is regression qualification, without a timing score.
