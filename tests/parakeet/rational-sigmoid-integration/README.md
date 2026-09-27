# Prepare the existing sigmoid tests for the repository suite

The frozen candidate tests include eight arithmetic/ownership facts and one
benchmark-only fact requiring Linux, runtime 10.0.8, CPU 2, vector width eight,
and product hashes from environment variables. The latter has already passed
in both normal and intrinsics-disabled correctness runs. Preserve those sources
and results; do not impose the VM environment on ordinary repository tests.

`prepare_tests.py` creates a separate integration fixture. All eight arithmetic
fact bodies and their assertions remain unchanged. Remove the VM-only fact from
this fixture, retain its exact source alongside the patch, and replace optional
helper parameters with two explicit forwarding overloads. This preserves caller
behavior and the repository's existing no-optional-parameters test unchanged.
No product file or root file is modified.

Run once from repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/rational-sigmoid-integration/prepare_tests.py

Review the fixture and patch under
`artifacts/parakeet-rational-sigmoid-integration-tests-20260927`. The eventual
root campaign must retain the original full test census plus these eight passing
facts in both normal and AVX512-disabled modes, independently bind every actual
product, runtime and CPU identity, compare all compiled product methods with
the measured candidate, and qualify the real package consumer. The separate
intrinsics-disabled arithmetic qualification remains retained. Integration still
requires all fresh graph and complete Pyannote application admissions.
