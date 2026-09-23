# Fixed warmed graph qualification

Independent follow-up to the inconclusive M43 graph comparison. M44 traces of
the unchanged GPT-2 baseline show product compilation through calls 345/350;
the old 60-call warmup precedes these transitions. The old result stays intact.

All eight cases and all three engines now use 600 fixed warmup calls and 180
measurements per fresh process. Order: current, candidate, ORT, ORT, candidate,
current. All 24 three-call numerical processes run first. Retain every one of
37,512 calls, 8,640 measurements and 72 setup intervals. Publish three fixed
60-call measured blocks per timing process as well as complete clocks.

The managed consumer changes only the two timing-count constants. Normal SDK
build and full method inspection must preserve every other instruction,
branch target, local, exception region, method flag and public declaration.
The native runner changes the same two constants. No profiler or overrides.

All original output/native/ownership checks remain. Every candidate array must
match selected bytes; fresh ORT scaled error must be <=1e-4. All 24 process
max/min controls must be <=1.10, all eight candidate/current ratios <=1.05.
Exact rational clock arithmetic gives equal process weights, with no trimming,
pooling, favorable block selection or unchanged score retry.

Use C:/Python313/python.exe -X utf8 -B with run.py prepare, stage, launch,
observe, collect, then audit.py. CPU 2 executes and CPU 0 monitors; preflight
12 GiB available/3 GiB tmpfs, RSS 8 GiB, 900 seconds/job, four hours/campaign.
Models, fixtures and existing binaries are reused with immutable hardlinks.
