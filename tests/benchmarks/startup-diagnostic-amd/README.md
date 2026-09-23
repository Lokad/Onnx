# GPT-2 startup diagnostic for the Parakeet release check

Two unchanged-baseline processes, 1,200 calls each, with full CLR events and
request markers. No candidate, native timing, admission score, forced collection
or runtime override. All original graph execution, output validation and native
limits remain unchanged. Every clock and emitted event is retained.

Use C:/Python313/python.exe -X utf8 -B with run.py prepare, stage, launch,
observe, collect, then audit.py. The M43 graph failure remains inconclusive;
this diagnostic cannot qualify a release or replace any scored clock.

CPU 2 executes; CPU 0 collects, monitors and exports. Preflight 12 GiB available
and 3 GiB tmpfs; RSS below 8 GiB, 900 seconds/job, output below 256 MiB/job and
512 MiB total. Reuse retained inputs through immutable hardlinks. No downloads.
