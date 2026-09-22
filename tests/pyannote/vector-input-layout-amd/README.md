# AMD vector input layout qualification and screen

Qualify raw and actual-model outputs in AVX2 and AVX512 before timing. Preserve
2648 raw cases,20 supplemental,10 invalid/alias cases and119823360 actual values.
Then run production,candidate,candidate,production in four fresh processes.

Each call repeats max(1,ceil(2^31/(M*C*KH*KW*OH*OW))) times per pass, at most128,
using immutable fixture shapes. Freeze the complete iterations.json before runs.
One complete warmup and three measured passes keep every iteration clock/hash.
Divide each call's measured sum by3*iterations, then sum108call means, preserving
original shape weights. There is no time calibration, adaptive stopping, trimming,
or threshold change. All512weight preparation clocks remain separate.

The actual fixture manifest has1,074iterations per pass:17,184call clocks across
four processes (4,296warmup and12,888measured). Twelve independent scorer tests
cover thresholds, geometry ceilings, unequal weighting and missing/duplicate data.

Exact controls remain max/min <=1.10 aggregate,<=1.20 every form. Selection remains
candidate/production <=0.90aggregate,<=1.05every eligible form. An admission only
permits product qualification, not an application or Microsoft ORT speed claim.

From root with C:/Python313/python.exe -X utf8 -B:

    tests/pyannote/vector-input-layout-amd/run.py prepare
    tests/pyannote/vector-input-layout-amd/run.py stage
    tests/pyannote/vector-input-layout-amd/run.py launch
    tests/pyannote/vector-input-layout-amd/run.py observe
    tests/pyannote/vector-input-layout-amd/run.py collect
    tests/pyannote/vector-input-layout-amd/audit.py

Freeze consumer, fixtures and tools. CPU2 before CLR, monitorCPU0. Require12GiB
available/3GiBtmpfs preflight,8GiBRSS bound,1GiBminimum available/tmpfs,1GiBoutput,
2GiBartifacts,900seconds per worker,four-hour campaign. Collect terminal owners.
Preserve temporary VM service masks and account lingering through the window.
