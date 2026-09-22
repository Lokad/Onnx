# Twelve-position weight reuse: fixed complete-call screen

Compare selected Core `3c2f16b0` with qualified candidate `3ca0a2a5` on the AMD
EPYC 9V74, CPU2, .NET 10.0.8. The exact product and LayerGraphs assemblies are
loaded unchanged. Both roles call the same prepared ordinary graph entry point.
The preceding unprepared direct-convolution control is not used for this trial.

Four fresh unprofiled processes run production, candidate, candidate, production.
Every process executes all 108 captured calls / fifteen forms in fixed order,
one full warmup and three measured passes. Each call repeats
max(1,ceil(2^31/work)), at most 128, where work comes only from immutable geometry.
The frozen schedule is 1,074 iterations per pass: 17,184 total call clocks,
4,296 warmups and 12,888 measured calls. Divide by each call's iteration count
before aggregation; all original call weights and fallback forms are retained.

Call timers include ordinary graph execution, caller assertions/dispatch
recording, scans, rentals, layout conversion, kernels and graph epilogues.
Hashing and journal IO are outside. Setup loads identical fixtures and helper
weight clones before timing. Warmup populates 108 independent prepared graphs,
retaining 63,258,624 bytes. Those are component-call graphs, not whole-model
residency. The fixture's separate prepared clones occupy 21,086,208 bytes.

Another 512 clocks report graph creation and actual weight preparation for
32 eligible nodes over four passes/process. These clocks are separate from
call timing; their boundary differs from earlier pure-helper preparation clocks.
Prepared bytes and hashes must match the existing qualified layout.

The unchanged exact-clock scorer requires every process control <=1.10 aggregate
and <=1.20 per form, candidate/production <=0.90 aggregate and <=1.05 for every
eligible form. All gates are mandatory. No trimming, sample exclusion, adaptive
warmup, unchanged retry or application/ORT speed claim follows from this screen.

From root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/spatial-weight-screen/run.py prepare
    tests/pyannote/spatial-weight-screen/run.py stage
    tests/pyannote/spatial-weight-screen/run.py launch
    tests/pyannote/spatial-weight-screen/run.py observe
    tests/pyannote/spatial-weight-screen/run.py collect
    tests/pyannote/spatial-weight-screen/audit.py

Preparation requires closed AMD numerical and generated-code qualifications,
runs twelve inherited scorer tests, and validates the new driver locally on
both exact products without taking timing samples. Target preflight and resource
limits remain unchanged. Collect only terminal owners and retain all journals.
