# One vector float Sigmoid candidate

This changes only `CPUExecutionProvider.Sigmoid` over measured M78 Core49901366.
The float loop uses the existing `MathOps.ExpVector` with portable vectors when
SIMD and the dense layout permit it. Scalar tails, disabled SIMD, doubles and
validation remain. No graph fusion, exponential coefficients or compiler flags
change. Root product and `BENCHMARK.md` remain the qualified release.

Prepare source once with `tests/parakeet/vector-sigmoid-source/prepare.py`.
Then use Python `C:/Python313/python.exe -X utf8 -B` from the repository root:

1. `tests/parakeet/vector-sigmoid-build/run.py prepare`
2. `tests/parakeet/vector-sigmoid-build/run.py stage`
3. `tests/parakeet/vector-sigmoid-build/run.py launch build`
4. Observe that build owner to terminal using `run.py observe build`.
5. `run.py collect build`, then `review.py build`.
6. Only after compiled review, `run.py launch capture`.
7. Observe that capture owner to terminal using `run.py observe capture`.
8. `run.py collect capture`, then `review.py capture`.

These filenames in steps 4–8 are relative to this directory. Never rerun a
completed operation or relaunch because an observation times out. Preserve any
failure; do not use the numerical results as a model/application score.

The build audit requires 3,276 unchanged Core methods, only Sigmoid changed,
all 697 Data methods unchanged, equal public surfaces and method flags, and
no warning beyond the two previously recorded nullable diagnostics. The exact
runtime products and hardware mode are checked inside each test process.

Twelve tests run normally and twelve with hardware intrinsics disabled. They
include the three retained encoder Sigmoid/special-value tests, vector tails,
2,048,769 finite/bit-pattern sweep values per mode, exponent boundaries,
reversed/sliced/broadcasted/offset layouts, ownership, doubles and refusal paths.
The existing float1e-6 bound is retained; scalar fallback and double results are
exact. Full native/model checks and performance qualification follow separately.

The offline AMD VM uses SDK10.0.204/runtime10.0.8, CPU2 for workers and CPU0 for
monitoring. Build/capture preflight is2GiB available and1GiB tmpfs, owned RSS<3GiB,
free memory/tmpfs>=1GiB, output<512MiB. Individual builds are bounded at180s and
test processes at300s. All dotnet commands disable the terminal logger.
