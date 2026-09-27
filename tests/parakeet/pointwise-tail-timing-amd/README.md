# One fixed test of remainder row sharing

Original candidate Core 7cac6788 has passed arithmetic qualification 558f2a52
and emitted-code review 3715408c. Compare it with qualified baseline 47984318.
Do not rebuild either product or include the ineffective source-order repair.

Use all 38 actual matrix geometries: rows 1024/2048, reduction 1024 and the 19
observed output widths. Add width 224 at both row counts as unchanged full-panel
controls. Buffers use the original deterministic finite generator; they are not
captured model activations. Every call must preserve baseline output bits and
guards with zero kernel allocation. Verify complete input/packed-buffer hashes
after every pass. Packing, destination restoration and checking are outside the
timer, which brackets only the same raw packed-kernel delegate in both products.
This boundary can test the kernel prediction, not establish an application gain.

Run current/candidate/candidate/current fresh processes, normal hardware mode,
five complete warmup and five measured passes each: 1,600 retained clocks.
Require all per-shape and corpus max/min ratios within and between processes
<=1.10. Require candidate/current<=1.05 for every shape including both controls,
and lower summed latency across the 38 actual shapes. For both row counts the
baseline 222-column call must be slower than 225 as in the scoped observation,
and the candidate must shrink that difference. Keep all 246 repeatability
controls and 43 gates; preserve failures without retrying or discarding clocks.

Reuse the existing consumer-only build, offline SDK and serial worker. CPU2
computes; CPU0 supervises. Enforce foreign CPU<=1% with the original accounting
(including its documented limits for vanished processes), 3 GiB owned RSS,
600 seconds per process, >=1 GiB free RAM/tmpfs and 512 MiB artifacts. No runtime
JIT flags, performance switches, panel-size sweep or instrumentation in the kernel.

From the repository root, prefix with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/pointwise-tail-timing-amd/test_audit.py
    tests/parakeet/pointwise-tail-timing-amd/run.py prepare
    tests/parakeet/pointwise-tail-timing-amd/run.py stage
    tests/parakeet/pointwise-tail-timing-amd/run.py launch build
    tests/parakeet/pointwise-tail-timing-amd/run.py observe build
    tests/parakeet/pointwise-tail-timing-amd/run.py collect build
    tests/parakeet/pointwise-tail-timing-amd/audit.py build
    tests/parakeet/pointwise-tail-timing-amd/run.py launch capture
    tests/parakeet/pointwise-tail-timing-amd/run.py observe capture
    tests/parakeet/pointwise-tail-timing-amd/run.py collect capture
    tests/parakeet/pointwise-tail-timing-amd/audit.py capture

Execute mutations once and observe the same owner to terminal. Numerical and
provenance checks are separate from component admission. A passing comparison
permits complete Parakeet correctness and the original matched application lane;
BENCHMARK.md changes only after full release qualification.
