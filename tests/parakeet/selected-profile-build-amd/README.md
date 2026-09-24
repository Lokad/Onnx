# Complete Parakeet profile consumer for the selected release

Copy Core `672e5f30` and Data `065b7a7f` unchanged. Adapt the previously qualified
SampledAudio consumer by replacing exactly its two embedded product hashes.
All other source, 162 compiled methods, implementation flags, public interfaces,
locals, branch targets, exception clauses and complete public-call markers stay
the same. Main has exactly two changed string operands; 161 methods stay exact.

Build the consumer and a diagnostic inventory reader on AMD only. The inventory
reader adapts the retained complete-method/flag reader to SampledAudio, removing
its irrelevant Core/Data special cases. Six serial jobs: SDK, reader restore,
reader build, consumer restore, consumer build, inventory. No product rebuild.
SDK 10.0.204, runtime 10.0.8, CPU2 workers/threads and CPU0 monitor. Builds use
`--tl:off --nologo -v minimal`; all inputs and the offline environment are pinned.

Run `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect` separately, then `audit.py`. Each preparation refuses an
existing namespace. Preserve failures; observe only the original PID/birth and
never observe after closure. Full source and compiled instruction review must
pass before a separately frozen complete-request capture.

Bounds: 12 GiB available / 3 GiB tmpfs preflight, 8 GiB RSS, 900 seconds/job,
1 GiB remaining memory/tmpfs and output/job, 2 GiB artifacts, four-hour campaign.
No runtime overrides, product edits, packing-budget changes or new ORT score.
M61 control `e9c296b8` is rejected and M59 is parked. This is fresh attribution
of the selected release, with all twenty clips and existing public checks.
