# Identical-binary Where repeatability control

This is a benchmark control for the new Parakeet dense scalar selection kernel.
Both nominal roles use the exact selected release: Core 672e5f30 / Data 065b7a7f.
No candidate binary is measured. A passing control only permits a later candidate
comparison with the same consumer, census, phases and batches. Failed controls
remain closed with all clocks retained; no unchanged retry is allowed.

The prerequisites are source a49c010c, build 1df68687, numerical closure edd4cf6e
and complete native review e77693cf. The candidate is numerically qualified;
its effect on application latency remains unknown. The selected root stays
source 81f75c38, with all 422 inputs verified before preparation.

Use every one of the 220 valid numerical cases, including all new boundaries
and run layouts. The first 122 cases keep the earlier screen's exact order and
partitions: target6 (actual captured float masks), other_uniform42 and fallback74.
The remaining cases form added98. These labels are fixed groups for reporting,
not assertions about which cases a later candidate can specialize. Also report
all220. All expected outputs and independent input construction are qualified.
The fifteen intentional errors remain covered by numerical qualification.

Construct all fixtures once. Each case owns its inputs, expected bits, output
array and retained clocks. First run 600 warmup samples for every case in order;
only after this whole pass completes run 180 measured samples per case in the
same order. The same generic Work<T>.Phase method executes both phases.
The entire workload therefore precedes any measurements; a particular JIT tier
is not assumed. There is no adaptive warmup, delay, forced GC, runtime override,
reflection or direct helper call in this path.

Each sample uses batch=max(1,min(1024,65536//max(1,output_elements))) complete
CPUExecutionProvider.Where calls assigning their returned OpResults into a
preallocated array. Timers include validation, allocation, selection and result
construction. Setup, independent oracles and metadata/ownership checks remain
outside timing. At each phase end every batch output must have exact bits,
shape and success metadata, distinct tensor objects and unchanged full input
stores. Hold a warmup output across the complete measurement pass; mutation of
a final output must preserve held outputs, other results and inputs.

Four fresh ordinary processes run current,candidate,candidate,current on CPU2
with CPU0 monitoring. These role names indicate positions only: both products
must have equal hashes at preparation, remote staging and audit. Use .NET 10.0.8,
SDK 10.0.204, normal AVX512/AVX2/FMA availability and no managed implementation
flags. One unchanged consumer build is copied into both runtime directories.

Normalize every measured sample by Stopwatch.Frequency and batch; average all
180 samples per case/process. Aggregates sum case means. Require max/min across
all four processes <=1.10 for each of the five aggregates and <=1.20 for every
case. Additionally, middle-pair mean / outer-pair mean must lie in [1/1.05,1.05]
for every case and aggregate. Exact rational arithmetic determines admission.
Any false gain or regression beyond five percent fails this identical-binary
control. Keep all 686,400 clocks (528,000 warmup and 158,400 measured) and 880
setups; verify journal order is the whole warmup pass followed by measurements.
No trimming, favorable block selection, missing samples or selective case gates.

Seven serial jobs use the existing offline feed. Bounds remain 12 GiB available
memory and 3 GiB tmpfs before each job, 8 GiB RSS, 1 GiB remaining memory/tmpfs,
900 seconds per job, four hours total, 1 GiB output per job and 2 GiB artifacts.
Closed-only deduplication precedes execution; no maintenance overlaps timing.
Models are not copied. No Windows build/inference or product edit occurs.

Freeze tools, then run C:/Python313/python.exe -X utf8 -B with:

    tests/parakeet/dense-scalar-where-stability-amd/test_score.py
    tests/parakeet/dense-scalar-where-stability-amd/run.py prepare
    tests/parakeet/dense-scalar-where-stability-amd/run.py stage
    tests/parakeet/dense-scalar-where-stability-amd/run.py launch
    tests/parakeet/dense-scalar-where-stability-amd/run.py observe
    tests/parakeet/dense-scalar-where-stability-amd/run.py collect
    tests/parakeet/dense-scalar-where-stability-amd/audit.py

Preparation refuses existing namespaces. Observe only an unclosed run and
collect only terminal PID/birth owners. Local evidence is under
artifacts/parakeet-dense-scalar-where-stability-amd-20260924; VM evidence is under
/dev/shm/lokad-parakeet-dense-scalar-where-stability-20260924. Technical validity
and stability_admitted are separate; neither means an application improvement.
The living plan is .agent/m59-parakeet-dense-scalar-where-20260924.md.
