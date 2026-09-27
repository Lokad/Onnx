# Diagnose the shared LSTM timing disturbance

The [fixed layout screen](../decoder-lstm-layout-results/timing-20260927.md)
remains rejected by repeatability. Its unchanged fallback workers consistently
slow on the second measured pass. Observe one original qualified baseline process
to distinguish runtime activity before changing the measurement or candidate.

Reuse its complete-call consumer, all 380 captured calls in original order,
five warmup/five measured passes, zero prepared-weight capacity and normal hardware
mode. Keep baseline Core `0d224bcf` and Data `a3745392`. No product is rebuilt.
Only the diagnostic consumer is compiled. Preserve every original output hash,
input immutability check, held output and allocation/timer boundary.

Add preallocated per-call telemetry with start/end Stopwatch timestamps, GC
generation counts and cumulative pause duration. Copy the existing allocation-free
EventSource markers. Attach the retained EventPipe collector before fixture setup,
record all runtime GC/JIT/loader events and sample-profiler events using the existing
provider set, and use the existing complete exporter and Speedscope converter.
Require 7,600 paired markers for 3,800 calls, one worker PID/thread, all raw events,
zero reported loss, and bounded correspondence between marker and Stopwatch clocks.

Tracing and added telemetry can perturb execution, including GC timing. This
process is diagnostic only: it cannot replace a score, revise the original screen,
or justify discarding slow calls. Attribution requires observed runtime activity
on the same clock. Keep unexplained residual variation explicit.

Use the original supervisor: CPU 2 worker, CPU 0 collector/monitor, 8 GiB free RAM
and 2 GiB tmpfs before work, at most 4 GiB aggregate owned RSS, 64 MiB per job,
256 MiB total artifacts and 300 seconds per job. The seven serial jobs are SDK
check, consumer restore/build, collector version, one capture, complete export and
stack conversion. Preserve all stopped jobs and failed captures without replay.

From repository root run `C:/Python313/python.exe -X utf8 -B
tests/parakeet/decoder-lstm-runtime-observation/run.py` with `prepare`, `stage`,
`launch`, then `observe`. After the original owner and children are terminal,
`collect` once and `audit.py` once. Root source and BENCHMARK.md remain unchanged.
