# AMD vector output epilogue qualification and screen

This is a distinct successor to the rejected direct spatial prototype. Only its
output transpose/epilogue changes; input layout, reductions and all eligibility
remain. First qualify raw AVX2, raw AVX-512, actual-model AVX2 and actual-model
AVX-512 in separate processes. Require every original Windows observation,
non-finite fallback/overflow/alias checks and all 119.8 million actual values.
Any qualification failure prevents timing.

Then use the exact preceding benchmark consumer, four fresh processes ordered
production,candidate,candidate,production. Preserve one complete warmup and three
measured passes over all 108 calls: 1,728 call clocks plus 512 separate preparation
clocks. Candidate width is sixteen channels. Timers include every conversion,
scan, scratch cost, epilogue, partial tile and the four fallback calls per crop.
No root product change or application/ORT timing occurs in this screen.

Keep exact gates: process max/min ≤1.10 aggregate / ≤1.20 each form; candidate
≤0.90 aggregate / ≤1.05 every eligible form. Never omit slow shapes or repeat a
failed candidate unchanged. Passing only admits subsequent product qualification.

From root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/vector-output-epilogue-amd/run.py prepare
    tests/pyannote/vector-output-epilogue-amd/run.py stage
    tests/pyannote/vector-output-epilogue-amd/run.py launch
    tests/pyannote/vector-output-epilogue-amd/run.py observe
    tests/pyannote/vector-output-epilogue-amd/run.py collect
    tests/pyannote/vector-output-epilogue-amd/audit.py

Artifact: `artifacts/pyannote-vector-output-epilogue-amd-20260922`.
Remote: `/dev/shm/lokad-pyannote-vector-output-epilogue-20260922`.
Freeze exact binaries, source, fixture arrays and seven gate tests. CPU2 before
CLR, monitor CPU0; 12 GiB available / 3 GiB tmpfs preflight, 8 GiB RSS,
1 GiB available/tmpfs/output, 2 GiB artifacts, 900-second workers, four-hour
campaign. Collect only terminal owners. Preserve temporary service masks and
account lingering through the exclusive window, restoring them at its end.
