# Packed projection phase diagnostic

This diagnostic investigates runtime compilation during repeated e5 projection
measurements. It is separate from public model timing and does not change
production kernels, runtime settings or benchmark acceptance thresholds.

The probe compares original/duplicate accumulation, copied accumulation, copied
overwrite with a clear, and the same overwrite without a clear. Every mode shares
the same pinned packed weights. Arithmetic, accumulation semantics, exceptional
values, slice guards, input preservation and repeated output bits are checked
before timing. The overwrite prototype accepts only full 32-column panels;
its copied column-tail code remains accumulating and is unreachable.

`generate.py` extracts the current twelve/eight-row composer and the two/three-row
remainder leaves. It produces the exact source copies used in the experiment and
records all sixty changed accumulator initializers. If the kernel layout changes,
review the extraction and its guards before using the probe again.

The corrected harness validates the panel's typed storage contract directly.
It preallocates each seven-sample list. These remove unnecessary JSON and
list-growth work that can queue background compilation into later measured calls,
even when that work was originally executed outside the stopwatch.

## Build and run

Build with SDK 10.0.204. Supply the already qualified core DLL explicitly; this
project never implicitly rebuilds the product. From the repository root:

```powershell
python -X utf8 tests/e5/projection-phase/generate.py --output artifacts/phase-new/kernels
dotnet build tests/e5/projection-phase/PhaseProbe.csproj -c Release -o artifacts/phase-new/probe-bin -p:FrozenCorePath=C:/Users/JoannesVermorel/code/Onnx/artifacts/production-defaults-v2-20260919/frozen/Lokad.Onnx.dll -p:KernelSourceDirectory=C:/Users/JoannesVermorel/code/Onnx/artifacts/phase-new/kernels --tl:off --nologo -v minimal
dotnet build tests/e5/projection-phase/collector/Collector.csproj -c Release -o artifacts/phase-new/collector-bin --tl:off --nologo -v minimal
```

The qualified core hash is
`7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`.
The collector uses DiagnosticsClient 0.2.510501 and TraceEvent 3.2.6.
No diagnostic library is added to the product.

Actual arithmetic requires AVX-512F/FMA and one logical CPU available to the
runtime. On the AMD Linux host, launch the worker on CPU 2 before runtime startup,
then attach the collector on CPU 0. For example, with freshly created output
paths and the built payload already transferred and hash-verified:

```sh
taskset -c 2 dotnet probe-bin/PhaseProbe.dll samples.json 0 gate > target.log 2>&1 &
target_pid=$!
taskset -c 0 dotnet collector-bin/Collector.dll "$target_pid" trace gate > collector.log 2>&1
wait "$target_pid"
```

The worker waits up to ninety seconds for `gate`; the collector creates it only
after attaching. Wait for the worker's `Waiting for gate` log message before
attaching if startup is slow. An untraced control uses the same worker after
creating a fresh gate file, with no collector. The order argument is 0–9; the
recorded causal diagnostics used fixed orders 0 and 5. An AVX2-only machine can
run `dotnet PhaseProbe.dll --host` for unsupported-ISA and packing checks, and
`--trace-smoke <gate> <output.json>` to qualify event capture without AVX-512.

## Evidence boundary

The probe retains all sixty isolated records and twenty 72-matrix bank records.
The bank contains 84,934,656 packed bytes. Each record has its complete original
warmup batch and seven measured batches, for 640 saved intervals per worker.
Each interval contains exact Stopwatch boundaries, process/thread CPU counters,
and before/after bounds around custom EventSource markers. The marker payload
uses explicit 4/4/8-byte fields; implicit numeric overload widening is unsafe
for this schema.

The external collector records runtime JIT, compilation, loader and GC events,
with timestamps, native thread IDs, method identities and optimization tiers.
It also records custom markers and event loss. Runtime event timestamps are
mapped to the worker clock using the intersection of marker emission bounds;
the auditor allows 100 ns of timestamp conversion/rounding uncertainty per side. Event loss,
missing/duplicate markers, wrong ticks and inconsistent clocks invalidate the
correlation. Background compilation duration includes scheduling effects and is
not itself CPU time. Foreground and process CPU counters provide separate evidence.

`audit.py --artifact <retained-campaign>` audits the first eight-worker diagnostic;
add `--confirmation` for the eight-worker normal-tiering scaffold comparison.
It requires the collected archive, deployment bundle and complete supervisor
records, checks all expected cases/intervals/flags and refuses existing outputs.
Do not rerun a successful immutable writer. Eight malformed-evidence checks are
built into the confirmation audit.

Traced wall times are diagnostics. Always retain untraced controls, all samples
and startup/warmup behavior. This tool neither attributes old untraced results
retroactively nor establishes calibrated full-model performance. See the
[recorded investigation](results-20260919.md) for the bounded findings.
