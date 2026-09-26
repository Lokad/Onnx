# One ORT-derived rational sigmoid candidate

Replace scalar exponential arithmetic for eligible float vectors with the fixed
rational logistic coefficients and order observed in ORT revision
2e2543fbe9fae542f921d47a72d21d5a4ef0b710. Preserve the qualified padding root as
baseline (Core 8bb22038, Data d02dbf55). Source preparation copies its 437 bound
inputs and changes only Sigmoid, adds one private non-inlined vector helper and
adds the nine retained sigmoid test facts. No release source changes yet.

Keep portable Vector<float>, whose observed AMD width must remain eight, output
allocation, graph structure and compiler settings fixed. The helper handles its
own remainder; the original zero-based scalar loop remains independent. No shared
exponential change, new ISA path, graph fusion, coefficient or unrolling search.
The helper includes Microsoft's MIT notice. Keep the historical vector rejection.

The compiled inventory must have 3,281 unchanged existing Core methods, only
Sigmoid changed, exactly one private helper added with NoInlining, and all 697
Data methods unchanged. Public surfaces, assembly metadata and every existing
method flag must match. No warning beyond the two retained CS8604 sites is allowed.

Reuse the existing 12 focused tests normally and with hardware intrinsics disabled.
The nine sigmoid facts cover 2,048,769 sweep values per mode, +/-18 saturation and
other rounding boundaries, special values, every vector tail, layouts, immutable
inputs, owned outputs, exact scalar/double behavior and runtime/product identity.
The only test-source addition verifies normal-mode Vector<float>.Count == 8.
Float tolerance stays 1e-6 and double tolerance stays exact for this contract.

During these correctness processes, allow targeted JIT disassembly, code bytes
and a log output filename. Retain every emitted version for Sigmoid and its new
helper. An optimized helper vector body must show nine FMAs, 256-bit operands,
division, no exponential call and no integer conversion. The public method must
call the helper; scalar mode must not enter it. Listings do not join tiers to
individual clocks. No timing from these instrumented tests is a performance score.

Build and contracts use the established serial supervisor: CPU2 for processes,
CPU0 for monitoring; SDK10.0.204/runtime10.0.8; 2 GiB available and 1 GiB tmpfs
before jobs, RSS <3 GiB, >=1 GiB free, <=512 MiB staged/output, 180 seconds/build
job and 300 seconds/test process. Offline package feed only. All .NET commands
disable terminal logging. Freeze source and tools before staging; preserve failures.

From repository root, prefix these commands with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/rational-sigmoid-source/prepare.py
    tests/parakeet/rational-sigmoid-build/run.py prepare
    tests/parakeet/rational-sigmoid-build/run.py stage
    tests/parakeet/rational-sigmoid-build/run.py launch build
    tests/parakeet/rational-sigmoid-build/run.py observe build
    tests/parakeet/rational-sigmoid-build/run.py collect build
    tests/parakeet/rational-sigmoid-build/review.py build
    tests/parakeet/rational-sigmoid-build/run.py launch capture
    tests/parakeet/rational-sigmoid-build/run.py observe capture
    tests/parakeet/rational-sigmoid-build/run.py collect capture
    tests/parakeet/rational-sigmoid-build/review.py capture

Only observe may repeat while the same owner is live. Collect and review once
after terminal ownership. Neither a timeout nor an incomplete observation permits
a restart. Audit stdout belongs outside the campaign folder. Do not edit any
frozen tool or completed evidence to repair reporting.

If contracts and generated-code checks pass, prepare a separate performance screen
using the unchanged 46-case consumer, census, rounds, process order and all original
gates: >=75% weighted saving, <=5% per-case regression, <=10% repeatability and
strict process-total separation. The qualified scalar root is the baseline.
Instrumented code-generation runs are separate from ordinary-runtime timing.
Passing the screen permits full model/application and release regression checks;
it does not establish the projected ~3.6% transcription gain or release admission.

Local: artifacts/parakeet-rational-sigmoid-build-amd-20260927.
VM: /dev/shm/lokad-parakeet-rational-sigmoid-build-20260927.
