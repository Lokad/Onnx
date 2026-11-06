# Paradee benchmark conditioning on .NET 11

The original 10-second conditioning period ended while default .NET tiering was
still compiling Lokad.Onnx methods. Fixed 90-second diagnostics on the existing
AMD VM showed latency stabilizing after roughly 20–30 seconds, with the last
core method loads between 18.1 and 23.2 seconds in the six traced processes.
No runtime override was used to disable tiering.

Fresh Paradee comparisons therefore require **at least 32 calls and 40 seconds**
of warmup for **both engines**, followed by the same 33 measured complete calls.
The final six-pair protocol, original exports and numerical bounds are unchanged.
Historical 10-second observations remain valid records of their stated protocol;
they must not be combined with newly conditioned process pairs.

## Selected runtime time courses

Each cell is the mean execution-plus-copy latency in milliseconds for the full
10-second window. These are descriptive time courses from one process per
configuration, not independent source/native comparisons or parity results.

| Workload | Collection | 0–10s | 10–20s | 20–30s | 30–40s | 80–90s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32,62tokens | untraced | 137.973 | 100.627 | 83.509 | 82.845 | 84.130 |
| INT8,14tokens | untraced | 101.017 | 41.725 | 37.556 | 37.145 | 37.575 |
| FP32,62tokens | traced | 167.260 | 109.995 | 90.483 | 90.651 | 89.784 |
| INT8,14tokens | traced | 99.082 | 53.661 | 42.307 | 42.031 | 42.201 |

The selected core is from `b680250d`, runtime source `6b920857`. The complete
[JSON evidence](paradee-conditioning-20261106.json) retains every window for all
12 predetermined runs, including a temporary Fourier prototype with its route
enabled and disabled in the same binary. That prototype had not been selected;
its time courses do not establish a source improvement.

## Scope and reproduction

The box is `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45. Each process uses CPU 2
and SDK/runtime 11.0.100-rc.3.26522.148/11.0.0-rc.3.26522.148. The inputs are the
committed 62-token FP32 voice 0/speed 0.8 and 14-token INT8 voice 0/speed 1.0 fixtures.
Each of the three configurations runs 90 seconds without tracing, then 90 seconds
with runtime events and thread-stack sampling. Every call copies the complete
waveform and checks finite shape, input immutability and held outputs.

The diagnostics time Reset, Execute and output copies. Their outside-clock
validation/hashing differs from the shared GraphBench caller, so their latencies
are not coordinator scores. Compilation timing and thread-stack sampling are
separate evidence: the latter samples threads regardless of CPU usage, as
described in the [dotnet-trace documentation](https://learn.microsoft.com/en-us/dotnet/core/diagnostics/dotnet-trace).
All six traces report zero lost events. Temporal overlap does not establish
which individual compilation caused a latency change.

For new source/native scores use the [Paradee coordinator](../../eng/bench.md#paradee-runtime-comparison)
with a fresh 40-second manifest and all six passing private-model facts. The
shared worker preserves every warmup/measured clock. Reserve 45 minutes for the
full comparison; its 48 timing processes alone require at least 32 minutes of
conditioning. Keep the complete invocation below 60 minutes. Do not choose a
favorable subset of these diagnostics or extend a cohort after inspecting it.
