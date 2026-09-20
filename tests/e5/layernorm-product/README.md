# Wider LayerNorm: actual product qualification

This lane exercises `LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT=0` and `1` in separate
processes. The actual library contains the sixteen-float final transform;
statistics, operation association and tails retain their existing arithmetic.
The switch is off by default. No performance/default decision is made here.

Sources were adapted from `tests/e5/fingerprint-product`, removing the old
fingerprint experiment's cache checks and graph mutations. Those closed sources
and results are unchanged. The focused plan is
`.agent/m2-layernorm-product-20260920.md`.

From the repository root, after committing the implementation and harness:

    C:/Python313/python.exe -X utf8 -B tests/e5/layernorm-product/prepare.py --artifact artifacts/e5-layernorm-product-20260920

Preparation archives committed sources and builds CLI, backend tests, tensor
tests and replay with SDK 10.0.204. It freezes every source/payload/model/input
identity, a common core DLL, job schedule and resource limits. The new 45
backend tests check both public APIs, vector/tail widths, independent scalar
references, bias/no-bias, input/parameter guards, exact in-place output, retained
results and exceptional values. Test totals are 3,174 backend and 342 tensor.
Hardware-specific skips must be retained in full TRX results.

Launch the frozen `qualification/run.py` with `--payload <payload-directory>
--assets <repository-root> --label windows` (or `amd`). Python must have psutil.
The supervisor uses logical CPU 0 and workers inherit CPU 2 before CLR startup.
Only child environments are cleaned of runtime overrides; the parent's
LOKAD_ROOT and unrelated work remain unchanged. Each worker has 600 seconds,
12 GiB process-group RSS and 1 GiB available-memory limits. Every observed
process birth and resource sample is retained; any failed job stops the lane.

Eight sequential workers per host run full backend/tensor suites, five e5
cases and all retained shared-model scenarios, off then on. E5 covers
8/30/padded128/128/512 tokens, Default/Memory and facade/explicit contexts,
three calls, missing-input failure, input and held-output preservation. Shared
models are DINOv3, ResNet50 and GPT-2 including carried decoder state. Every one
of 60 e5 and 106 shared arrays per setting must pass the unchanged native
scaled 1e-4 gate; all off/on output bytes must be identical.

An additional ninth AMD job disassembles the actual product kernel, using
only declared JitDisasm/JitStdOutFile flags. It repeatedly executes both public
paths with a tail width, bias/no-bias, for at least 128 pairs and three seconds.
`code_audit.py` requires an optimized actual product body, double zmm transform,
original ymm arithmetic, narrowing from zmm doubles and no fused arithmetic.
This instrumented job supplies no latency ratio.

After all processes exit, `collect.py --artifact <local-artifact>` verifies
remote frozen files and actual terminal PID/creation-time pairs, transfers an
exact archive inventory and verifies every extracted file. Do not rerun a
successful collector. Independently audit each host:

    C:/Python313/python.exe -X utf8 -B tests/e5/layernorm-product/audit.py --payload <payload-or-collected> --assets . --label <windows-or-amd> --output <new-audit.json>

Retain failed attempts separately. Never overwrite successful inference,
collection or report outputs. Source, build and runtime identities, full tests,
raw arrays, independent audit and terminal process proof are needed before a
result is reported. Existing complete-model timing and audio numerical gaps
are unaffected by this correctness lane.
