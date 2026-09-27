# Runtime compilation occurs inside measured pointwise calls

The unchanged candidate's diagnostic trace contains an optimized recompilation
of the packed matrix kernel **inside a measured call**. This establishes that the
component measurement region is not free of compilation. The earlier untraced
screen remains not admitted; these events cannot retrospectively correct it or
establish an application speedup.

Candidate Core `7cac6788` runs the same forty shapes, finite buffers and five
warmup/five measured passes. Only the consumer gains outside-timer markers and
preallocated GC/clock telemetry. The three raw timer/kernel statements are
unchanged. All 400 outputs match the original screen's exact hashes, with all
input/packed immutability, guard and zero kernel-allocation checks preserved.

The trace audit reconciles **27,814 events, 400 calls and 800 markers**, with zero
reported event loss and one clock-offset intersection 3.593 microseconds wide.
Every marker belongs to worker PID/thread 1224499. All seven jobs complete with
code zero; peak owned RSS is 502,767,616 bytes and capture takes 6.49 seconds.
The collector and supervisor run on CPU 0; all worker threads remain on CPU 2.

There are 285 complete JIT compilation intervals and four complete GC intervals.
In the fifth measured pass, call 360 (`M=1024, N=1024, K=51`) encloses a
2.236513 ms `OptimizedTier1OSR` compilation of
`MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump`. OSR means the runtime
compiles optimized code while the method is already executing. That call lasts
4.368948 ms; the same shape's preceding four measured calls last 1.903278,
2.026115, 1.986124 and 1.998513 ms. Keep all five clocks.

Seventeen compilation intervals overlap measured kernel calls. Their union
overlap is 2.236513 ms on the worker thread and 22.780678 ms on a background
thread sharing CPU 2. Background work includes the matrix kernel, array copying,
output comparison and enumeration. Those other operations are outside the raw
timer in source, but their asynchronous compilation can overlap it. These are
elapsed intervals, not measured CPU time or additive recoverable latency.

No GC interval or GC suspension overlaps the measured kernels; every measured
GC counter delta is zero. Other runtime suspensions (`SuspendOther`) overlap by
2.540273 ms while the sample profiler is enabled; that reason does not identify
every cause, and these suspensions must not be reported as GC collections.
The [complete attribution](pointwise-tail-runtime-observations-20260927.json)
includes all forty shape records, every phase and the exact compilation intervals.

The diagnostic does not explain every variation. Eleven shapes still exceed a
1.10 max/min ratio, and the slowest aggregate measured pass has no observed JIT
overlap. Tracing changes execution: the pronounced shape/pass differs from the
untraced screen. Neither compilation nor profiler overhead can be subtracted
from the original clocks, and the failed 222-versus-225 prediction stays failed.

Keep this one candidate unchanged. Next, run the original complete Parakeet
correctness checks in normal and AVX512-disabled modes, then make an independent
decision using the existing matched current/candidate/ORT application protocol.
Retain its original repeatability, >=1% corpus-gain and <=5% per-clip-regression
gates. This proceeds on a diagnosed mechanism and a numerically qualified
candidate, without asserting an isolated speedup. The qualified release remains
**1.188× ORT** until a complete application comparison and release checks pass.

Capture artifact: `artifacts/parakeet-pointwise-tail-runtime-observation-amd-20260927`,
closure `47310420`; attribution artifact:
`artifacts/parakeet-pointwise-tail-runtime-analysis-20260927`, closure `75f3b259`.
Supervisor 1224433 / birth1790532367.32 and all children are terminal. The 207-file
collection and original failed timing evidence are preserved.
