# One public-call test of the observed Parakeet projection

The qualified decoder owns packed weights but executes the row-major path for
its final 640-by-8198 projection. The candidate changes that addressing only.
Its Core is af19b3b4; current is 65f15a41. Exact contracts are closed at fc00688c.
This is a fixed candidate comparison, not a kernel or runtime-flag search.

Time the allocating public `Tensor<float>.MatMul` call, including dispatch,
metadata, arithmetic and output ownership. This boundary excludes graph and
provider wrappers and buffer pooling. It is not graph-node wall time or an
application score. Import and preparation are outside timing and their products
are checked. Reflection attaches the existing map once during setup, never in
the measured call. No product is rebuilt or instrumented.

Six cases are fixed: captured final projection with its original map (target),
the same operands without a map, Scalar and Simd options, then the prediction
640-by-640 and encoder 1024-by-640 width guards. The last two use original model
weights and explicitly synthetic deterministic activations; they are fallback
controls, not captured node executions. The other four use the retained exact A.
One unpacked reference call per case occurs during setup. The mapped target and
unmapped reference must match the original captured output hash.

Use the unchanged vector-sigmoid screen build/capture supervisor and transport.
Build one Screen.dll against current; swap only Core between the two runtimes.
Run current/candidate/candidate/current as four fresh ordinary .NET 10.0.8
processes on CPU2, SDK 10.0.204, no runtime switches, no forced GC. Every case
performs 600 warmup rounds before any measurements, followed by 180 measured
rounds, in census order. Batch size is min(256, floor(65536/output length)), at
least one: seven wide calls, 102 narrow calls per round. Retain all 4,680 clocks
per process, allocations, copy and scratch counters, setup durations, raw
resource samples and process ownership. Exact bytes are checked for every output
in rounds 0, 599 and 779; every call's result type and length are checked. Held
outputs remain independent and inputs and prepared weights remain immutable.

Admission requires target mean candidate/current <=0.75, each unchanged fallback
mean <=1.05, each same-product case repeat max/min <=1.10, bit-exact outputs
across products and no increased copy or scratch. Means use all 180 measured
rounds and both processes equally; no trimming. Allocation increase is recorded
and included in the complete-call clock. Contract checks already found 104 extra
bytes per captured candidate call; no metadata optimization is bundled here.
Freeze cases, counts, consumer and score code before timing. Preserve any failed
gate; diagnose its cause before another experiment. A passing screen only permits
the predeclared complete twenty-clip application check (>=1% gain, <=5% per-clip
regression, original repeatability and numerical/decision/ownership gates).

Run from repository root using `C:/Python313/python.exe -X utf8 -B` with:

    tests/parakeet/decoder-packed-row-screen/test_score.py
    tests/parakeet/decoder-packed-row-screen/run.py prepare
    tests/parakeet/decoder-packed-row-screen/run.py stage
    tests/parakeet/decoder-packed-row-screen/run.py launch build
    tests/parakeet/decoder-packed-row-screen/run.py observe build
    tests/parakeet/decoder-packed-row-screen/run.py collect build
    tests/parakeet/decoder-packed-row-screen/audit.py build
    tests/parakeet/decoder-packed-row-screen/run.py launch capture
    tests/parakeet/decoder-packed-row-screen/run.py observe capture
    tests/parakeet/decoder-packed-row-screen/run.py collect capture
    tests/parakeet/decoder-packed-row-screen/audit.py capture

Observe until terminal before collecting; never replay completed commands.
The local namespace is artifacts/parakeet-decoder-packed-row-screen-v2-amd-20260927;
VM namespace /dev/shm/lokad-decrow-screen2-20260927. A 32 MiB campaign limit and
small collections fit the last 210 MB of the 50 decimal GB allocation budget.
Keep all canonical models and prior evidence. BENCHMARK.md stays qualified until
the application and broader release checks pass.

The first namespace is preserved at failure a5d10595. Its consumer built but the
frozen zero-warning audit rejected CA1416: an assertion helper around the Linux
test did not satisfy platform analysis. No timing ran. V2 changes only that
consumer guard to an explicit throw and binds the original source, failed build,
score code and identical census. The two Core products remain unchanged.
