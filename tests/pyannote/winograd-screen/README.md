# Complete-call screen for the qualified Winograd prototype

This protocol is fixed before timing. Numerical closure `8aae1968`, codegen
closure `28c8d946` and independent review `1d1b2d66` admit this screen.
The exact qualified component binary remains
`937ab50e8d9cc140d7c27ba9d010bd8fe32638bc115173e436ef44a9646edd84`.
A separate consumer binds its existing private methods through typed delegates;
it does not recompile or change either arithmetic implementation.

Both roles time output allocation, scratch planning/rentals/returns, validation,
finite checks, runtime transforms, convolution, layout conversions and the
same bias/residual/ReLU epilogue. They return an independently owned array.
Input loading, weight preparation, hashing, assertions on returned hashes and
journal serialization are outside the call timer. All preparation clocks are
reported separately. This is a standalone convolution-component boundary;
it excludes graph scheduling and the audio application and supplies no ORT timing.

All 87 eligible captured stride-one calls and their eight forms remain in
their original order and multiplicity. Iterations use the existing geometry
rule ceil(2^31 / direct multiplication count), giving exactly three per call.
Each process executes one warmup and three measured passes: 1,044 calls,
261 warmup and 783 measured. Four fresh processes run current, candidate,
candidate, current: **4,176 clocks**, including **3,132 measured**. All 29
weight sets are prepared before each pass, yielding **464 preparation clocks**
across the four processes. No calibration, trimming or adaptive repetition.

Exact clock fractions require candidate/current <=0.90 aggregate and <=1.05
for all eight forms. All 18 repeatability controls must pass: max/min process
means <=1.10 aggregate and <=1.20 per form, separately for each role. Both
candidate aggregate process means must be strictly below both current means.
Every candidate output must match its own qualified numerical hash; every
current output must match the selected direct hash. Different finite rounding
between roles was explicitly qualified under the unchanged native error bound.
All read-only, prepared-weight and held-output checks remain mandatory.

Before any scored worker, normally build the new consumer and run one separate
verification process through both roles: 174 complete calls and held outputs.
Failure stops the campaign before timing. Build and verification clocks are
diagnostic and never scored. SDK10.0.204/runtime10.0.8, CPU2 and ordinary AVX512;
no profiler, ISA, inlining or tiering overrides during this campaign.

Eight jobs: SDK identity, restore, Release build, verification, current-a,
candidate-a, candidate-b, current-b. Build preflight10GiB available; verification
and timing12GiB. Tmpfs preflight3GiB; live8GiB owned RSS ceiling,1GiB available
and tmpfs floors,900seconds/job,1GiB output and2GiB artifact limits. Supervisor
CPU0. Freeze every source, fixture, runtime, interpreter and dependency before
launch. Retain all failures; no unchanged timing retry or partial-result selection.

Use `C:/Python313/python.exe -X utf8 -B tests/pyannote/winograd-screen/run.py`
with `prepare`, `stage`, `launch`, `observe`, then terminal-only `collect`.
Run `audit.py` afterward and publish every clock. An admitted screen permits
separate product integration work and full qualification; the existing3%
complete-application improvement gate and all native/public-result/long-meeting
regression checks still determine whether source is integrated.
