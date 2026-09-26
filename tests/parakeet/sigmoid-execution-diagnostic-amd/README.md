# Diagnose the existing scalar and rejected vector sigmoid implementations

The current complete profile attributes 2.459 seconds to public sigmoid calls.
ORT's exact executed AVX512 SiLU uses rational logistic arithmetic; the rejected
managed trial uses general vector exponential. Its vector width, generated code
and allocation differences were not observed. This diagnostic resolves that gap
before selecting another implementation. It produces no performance admission.

Use actual qualified Core 8bb22038 and retained rejected Core 9bae182b, unchanged.
The existing compiled inventories verify the scalar method, ExpVector, allocation,
options and view helpers against the previous screen's baseline. Unrelated product
methods are not claimed equal. Preserve the old failed screen aaca2b2f.

Reuse every original fixture, value, call, 600 warmup / 180 subsequent rounds,
four-process order, numerical and ownership check. The exact source is recovered
after removing six explicit instrumentation edits. Build one common consumer on
the VM; the only runtime-file difference is the product DLL. No product build.

Allow exactly two diagnostic settings:

    DOTNET_JitDisasm=Lokad.Onnx.CPUExecutionProvider:Sigmoid Lokad.Onnx.MathOps:ExpVector
    DOTNET_JitDisasmWithCodeBytes=1

Record thread allocation bytes and GC collection counts around each existing
timed batch, with both counter calls outside its timer. Preserve all 143,520
clocks and counters, including every warmup. Retain every emitted native body and
tier, helper call, vector width and stack operand. Listings establish generated
code; they do not associate a tier with every clock or establish historical tiers.
Allocation-counter equality excludes added bytes, not allocation latency or GC cost.

Decision: if large dense calls allocate the same payload in both products while
the emitted optimized vector loop has substantial conversion/call/spill overhead,
investigate that arithmetic/code mechanism against ORT's observed rational leaf.
If materialization or added allocation differs, diagnose that path first. Inspect
scalar-option and empty branches independently: their regressions cannot be
explained by vector polynomial work. Keep inconclusive findings inconclusive.
Do not choose a new variant from these clocks alone or alter an old failure.

The original worker, monitor and whole-record validator are reused. CPU2 computes;
CPU0 monitors. Build: 2 GiB available / 1 GiB tmpfs, 3 GiB RSS, 180 seconds/job.
Capture: same memory bounds, 900 seconds/process, 1 GiB remaining and 512 MiB total
output. These are the existing small operator-screen limits, not model-profile
limits. All work is serial; verify idle owners and immutable inputs before launch.

Prefix with `C:/Python313/python.exe -X utf8 -B` from repository root:

    -m unittest discover -s tests/parakeet/sigmoid-execution-diagnostic-amd -p test_*.py
    tests/parakeet/sigmoid-execution-diagnostic-amd/run.py prepare
    tests/parakeet/sigmoid-execution-diagnostic-amd/run.py stage
    tests/parakeet/sigmoid-execution-diagnostic-amd/run.py launch build
    tests/parakeet/sigmoid-execution-diagnostic-amd/run.py observe build
    tests/parakeet/sigmoid-execution-diagnostic-amd/run.py collect build
    tests/parakeet/sigmoid-execution-diagnostic-amd/audit.py build
    tests/parakeet/sigmoid-execution-diagnostic-amd/run.py launch capture
    tests/parakeet/sigmoid-execution-diagnostic-amd/run.py observe capture
    tests/parakeet/sigmoid-execution-diagnostic-amd/run.py collect capture
    tests/parakeet/sigmoid-execution-diagnostic-amd/audit.py capture

Freeze tools at preparation. Only observe repeats while original owners are live.
Collect/audit each phase once after terminal owners; keep audit stdout outside the
campaign directory. Never repeat completed workloads to repair reporting.

Local: artifacts/parakeet-sigmoid-execution-diagnostic-amd-20260927.
VM: /dev/shm/lokad-parakeet-sigmoid-execution-diagnostic-20260927.
