# One runtime observation of the unchanged pointwise candidate

The fixed shape comparison remains not admitted. Observe candidate Core
`7cac6788` once with exactly the same forty shapes, deterministic finite buffers,
five warmup/five measured passes and raw-kernel timer. No product is rebuilt and
no timing result from this diagnostic replaces the original screen.

Only the consumer gains preallocated telemetry, allocation-free EventSource
markers outside the original kernel timer, and a collector-ready handshake before
fixture setup. Preserve every output bit, input/packed hash, guard and zero kernel
allocation check. Also require the output hashes retained by the original screen.

Reuse the existing EventPipe worker, provider set, exporter and stack converter.
Require 400 calls, 800 markers on one native thread, zero reported event loss and
one clock-offset interval consistent with all marker brackets. Inspect JIT, GC,
runtime suspension and sampled stacks; retain all unexplained variation. Tracing
can change runtime behavior, so elapsed event overlap is neither CPU time nor an
amount that can be subtracted from the untraced clocks.

CPU 2 runs the consumer; CPU 0 runs the collector and supervisor. Preserve the
existing bounds: 8 GiB available RAM / 2 GiB tmpfs preflight, 4 GiB owned RSS,
64 MiB per job, 256 MiB total artifacts and 300 seconds per job. The seven serial
jobs check the SDK, restore/build only the consumer, check the collector, capture
once, export every event, and convert stacks. Reuse retained products and tools.

From repository root, prefix commands with `C:/Python313/python.exe -X utf8 -B`:
`tests/parakeet/pointwise-tail-runtime-observation/run.py prepare`, then `stage`,
`launch`, and `observe`. Collect once after that owner and its children terminate;
run `audit.py` once. Preserve any failed run without replay. Attribution is a
subsequent read of retained evidence; root source and BENCHMARK.md stay unchanged.
