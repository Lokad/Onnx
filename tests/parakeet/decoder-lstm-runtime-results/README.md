# Runtime diagnosis of the LSTM screen

**The short component warmup did not finish JIT tiering.** The diagnostic trace
contains compilation inside its declared measurement region, including compilation
of the executing LSTM method. This does not turn the original failed screen into
a pass.

The capture uses the qualified baseline, zero prepared-weight capacity, the same
380 calls, five warmup/five measured passes, and normal hardware policy. All 3,800
calls preserve their outputs (11,400 arrays). All 7,600 markers and 76,708 exported
events are retained with zero reported loss. One clock offset satisfies every
start/end bracket, with a 0.002771 ms range and the declared one-microsecond
conversion tolerance. Capture closure is `a00f9195`; the separate read-only
attribution closure is `dfa4cef5`.

The slowest traced measured pass takes 1.426124 seconds; the other four take
1.190467, 1.194302, 1.189984 and 1.164900 seconds. During the slow pass:

| Observation | Elapsed overlap with actual calls |
|---|---:|
| Background compiler thread, sharing CPU 2 | 697.542 ms |
| LSTM compilation on the calling thread | 34.126 ms |
| Background GC interval | 14.154 ms |
| Nonconcurrent GC interval | 1.463 ms |

Compiling an optimized LSTM while that method is executing—on-stack replacement—is
wholly inside one recorded call. In total, 569 compilation intervals overlap the
slow pass, including graph execution, input resolution, node dispatch and LSTM.
The region includes changing compiled code and competition for its sole CPU.
These overlaps are elapsed intervals, **not CPU measurements or additive savings**.

GC pause counters add only 4.007 ms during that pass. Events named
`GC/SuspendEE...` also include `SuspendOther` from sampling; those events must not
all be described as garbage collection. The raw collection and suspension census
is retained separately from JIT intervals.

Instrumentation moved the peak from the original second measured pass to the
first. The trace cannot assign exact costs to the old untraced clocks, and
prepared-path variation between processes remains unresolved. The original screen
stays rejected, with every clock and threshold unchanged.

The next decision concerns complete transcription. Keep the one ORT-guided layout
candidate unchanged; first run complete Parakeet correctness, then the existing
independent current/candidate/ORT application comparison. Preserve its warmups,
all samples, >=1% total gain, <=5% per-clip regression and original repeatability
and numerical gates. No isolated LSTM speedup or universal fallback equivalence is
claimed. Only that application verdict can select this candidate for broader
release qualification. This replaces further component tuning with a decision at
the user's actual workload boundary.

[Full attribution, identities and phase totals](runtime-20260927.json).
