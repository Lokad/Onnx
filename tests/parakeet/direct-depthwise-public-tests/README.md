# Portable depthwise regression tests

The isolated eight-test fixture loads a pinned historical DLL and uses VM
environment variables for identity and geometry evidence. Those requirements
cannot enter a clean-checkout public test suite.

`prepare.py` preserves six qualified behavioral test bodies and embeds all 59
geometries from the closed normal/scalar runs. It removes the VM identity fact
and external file plumbing, yielding seven ordinary facts. Inputs, borders,
vector tails, special values, unsupported options, logical views, pooling and
ownership remain covered. All exact-bit assertions are retained.

The portable oracle is the existing generic convolution with
`MaxDegreeOfParallelism=2`, which excludes the direct path by its entry condition.
Its operands use the same materialization as the earlier oracle. This is a
different public regression oracle; the separate cross-assembly qualification
against M78 remains intact and must not be described as replaced or rerun.
The scratch expectation also checks FMA support, matching the direct entry guard.

Run from the repository root with
`C:/Python313/python.exe -X utf8 -B tests/parakeet/direct-depthwise-public-tests/prepare.py`.
Preparation verifies the original source and qualification identities and refuses
existing outputs. The result is source only. Before root integration, compile
and execute the portable fixture on the VM, then require complete root suites,
compiled product equivalence and all model/application release gates. No current
VM campaign, product source or frozen proof is changed by this preparation.
