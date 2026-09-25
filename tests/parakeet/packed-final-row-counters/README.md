# Confirm that final-row reconstruction disappears on the real corpus

Require completed full-model and public-corpus correctness plus censusfa159300
before preparing this stage. Products remain selectedM73 and candidateM78
Core49901366/Data01e9e784. Build only the existing counter consumer with one
prospective assertion change: every observed feed-forward CopyY count must be
zero. No product or arithmetic source changes.

Preserve all20 clips, actual matrix dimensions,2856 ordered profile records,
96 feed-forward calls per request, immutable inputs, independently held outputs
and exact selected/candidate frontend and encoder hashes. The new route must
avoid the same1740 transient16MiB packs, while reconstructing zero matrices.
Require29,192,355,840 fewer cumulative scratch bytes and zero additional copy
bytes per corpus. These are traffic counts, not retained memory or time savings.

Run selected/candidate in separate processes in normal and AVX512-disabled modes.
Retain11GiB available RAM,2GiB tmpfs,12GiB RSS,900seconds per worker,1GiB remaining
RAM/tmpfs,128MiB output,CPU2 workers andCPU0 monitoring. Before each worker,
record a bounded900-second wait for the unchanged memory threshold.

Use Python3.13 -X utf8 -B with run.py prepare, stage, launch build, observe build,
collect build and review.py build. After the zero-warning consumer review passes,
launch capture, observe capture, collect capture and review.py capture. Freeze
tools before staging, verify terminal PID/birth identities, and preserve failures.
The matched full-application comparison and retained release controls follow.
