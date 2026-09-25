# Observe the two failed e5 controls

This is a local draft. No VM campaign has been prepared or run.
The active Parakeet application comparison owns the VM until it is closed.

Keep the exact selected f95a13c5 and candidate 49c3a958 Core binaries, the
original d827e3b9 benchmark loop, both failed e5 inputs and all 780 calls.
Eight fresh processes will run selected, candidate, candidate, selected for
e5-8tok, then the same order for e5-512tok. CPU 2 executes; CPU 0 records
compilation and collection events. No new performance score is produced.

`ClockProbe.cs` changes only the two accepted input keys from the previous
diagnostic. The original four observation hooks remain outside the timer.
`ExportAll.cs.txt` changes only the output stream to lossless gzip; all event
fields, raw bytes, ordering and summary counts remain unchanged. Before new
capture, export one retained trace and require its decompressed bytes to match
the original complete event export exactly.

Run `C:/Python313/python.exe -X utf8 -B` on `review_reuse.py` to verify source
recovery, consumer/product provenance and unchanged compiled/event audit logic.
Transport, independent audit and publication drafts now exist. After the active
Parakeet application closes, use `run.py prepare`, `stage`, `launch`, `observe`,
`collect`, then `audit.py`. Do not repeat completed collection or publication.
The governing ExecPlan is `.agent/m76-e5-repeatability-diagnostic-20260925.md`.

Preparation verifies all prior closed results and records a storage estimate
from complete retained traces. The unchanged 512 MiB stage bound remains hard.
This estimate uses the earlier 30-token input; it does not guarantee the two new
trace sizes. Stage only with 11 GiB plus the full 512 MiB stage allowance available,
and 3 GiB free tmpfs. This extra initial headroom accommodates retained traces;
every worker still uses the original 11 GiB preflight and all original limits.
If headroom is insufficient, reclaim only verified obsolete duplicates while
the VM is idle. Never clean up during the active Parakeet score.

The consumer build is followed immediately by compiled instruction review.
The exporter build must reproduce the complete retained export before any
new capture. The final audit checks all 6,240 calls and 12,480 markers, with
every original correctness assertion and all raw events retained. Publication
uses `../e5-repeatability-diagnostic-results/publish.py`; interpretation follows
the actual observations. No diagnostic has been prepared or launched yet.

The old failed comparison and every clock remain retained. This observation
may explain runtime mechanisms; it cannot retroactively assign the original
timing difference to one cause or justify a favorable trimmed score.
