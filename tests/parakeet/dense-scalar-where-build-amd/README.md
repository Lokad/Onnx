# Dense scalar Where isolated build

Build frozen sourcea49c010c from selected81f75c38 without editing root product.
The sole existing-method change is CPUExecutionProvider.Where's Float arm:
the exact229instruction provider body is reconstructed prospectively, preserving
all old null/option/dtype validation, switch targets, nonfloat branches, flags
and public interfaces. The minimum4096/scalar guard matches M57; only the new
helper target differs. Generic Tensor.Where and3188otherCore/697Data methods
must remain exact, without compiler-generated identity renaming.

Two new internal/private methods are allowed: DenseScalarWhere.Try<T> and
SelectMixed<T>, both NoInlining|AggressiveOptimization. They introduce the new
algorithm, so this build proves their signatures/flags/scope, not numerical
semantics or equivalence to the old uniform-only body. Full numerical and native
code qualification is mandatory afterward. Seven adversarial scope tests cover
old instructions/edges/locals/flags/API, unrelated drift and new-method census.

The retained four-job controller performs SDKversion, offline restore, normal
ReleaseCLIbuild and complete Bridge method inventory on AMD. SDK10.0.204,
runtime10.0.8,CPU2compute/CPU0monitor,all .NET flags --tl:off --nologo -v minimal.
No Windows build/inference. Preflight12GiBavailable/3GiBtmpfs,8GiBRSS,
1GiBremainingmemory/tmpfs,900s/job,fourhours/campaign,1GiBoutput/job,2GiBtotal.

Freeze tools, then run C:/Python313/python.exe -X utf8 -B with
 tests/parakeet/dense-scalar-where-build-amd/test_checks.py;
 run.py prepare,stage,launch,observe,collect; then audit.py.
Refuse existing namespace:artifacts/parakeet-dense-scalar-where-build-amd-20260924;
VM:/dev/shm/lokad-parakeet-dense-scalar-where-build-20260924.
Collect only terminal PID/birth owners. Source/contracts remain pinned; no model
copies, network package restore or new dependency. Follow
.agent/m59-parakeet-dense-scalar-where-20260924.md for all subsequent gates.
