# Qualify the single transpose dispatch candidate

Build the isolated 447-file snapshot from transpose-axis-source, retaining the
actual qualified Data binary. Compare every Core/Data method, implementation
flag, public declaration and assembly attribute against root ee4a38ff. Only
Tensor<T>.TransposeInto may change; all arithmetic leaves stay identical.
Preserve the four existing compiler-warning occurrences.

Run 38 backend contracts in each normal, disabled-AVX512 and disabled-intrinsics
mode. These include the six new bit/layout/ownership facts, all existing
TransposeBlockAgreement/Boundary/FastPath tests and actual product/hardware
identity. Run 84 tensor contracts in each AVX-capable mode and 83 without
intrinsics: all existing face-dispatch, face-vector, reshape/transpose and
source-policy tests. The raw AVX 8-by-8 leaf test runs in the first two modes;
it requires AVX and is excluded only from the disabled-intrinsics process.
All public transpose contracts run in every mode, including all 76 observed
family/geometric combinations. No inference or performance score is produced.

Reuse the established supervisor, SDK 10.0.204/runtime 10.0.8, offline feed,
CPU 2 for work and CPU 0 for monitoring. Keep 2 GiB available/1 GiB tmpfs
preflight, 3 GiB RSS, 1 GiB minimum free, 512 MiB output, 180-second build jobs
and 300-second test jobs. No limits or audit rules are relaxed.

From repository root, prefix each command with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/transpose-axis-source/prepare.py
    tests/parakeet/transpose-axis-build/run.py prepare
    tests/parakeet/transpose-axis-build/run.py stage
    tests/parakeet/transpose-axis-build/run.py launch build
    tests/parakeet/transpose-axis-build/run.py observe build
    tests/parakeet/transpose-axis-build/run.py collect build
    tests/parakeet/transpose-axis-build/review.py build
    tests/parakeet/transpose-axis-build/run.py launch capture
    tests/parakeet/transpose-axis-build/run.py observe capture
    tests/parakeet/transpose-axis-build/run.py collect capture
    tests/parakeet/transpose-axis-build/review.py capture

Freeze source/tools before preparation. Mutation commands write once; observation
is read-only. Verify every owner is terminal before collection. Preserve any
failed attempt and diagnose retained evidence before recovery. Root product
stays unchanged. Complete original public results and the independent original
application comparison remain necessary before broader release qualification.
