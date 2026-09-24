# Retained root source-policy failure

The first normal-root qualification is **rejected**. Its tensor suite passed
367 cases and failed `NoNullForgivingTests.SourceTree_HasNoNullForgivingOperators`:
the new dense-mask helper contained `output = null!`. The existing source policy
requires nullable annotations and flow analysis instead of suppression.

The preceding builds and compiled inventory passed. All 3,253 Core and 697 Data
method bodies, flags and public declarations matched the measured candidate.
The full backend suite passed 3,499 cases and skipped its expected 41, including
passes for all 28 new masking cases. Every other tensor outcome matched the
previous release. Six later suite/package jobs were not executed. These partial
successes do not qualify the product.

All ten executed jobs, logs, TRX files, source inputs and resource observations
remain in the closed failed campaign. Every owner is terminal. No test result
was removed or reclassified, and the policy test is unchanged.

The separate successor applies the library's established nullable-output
contract: `[NotNullWhen(true)] out Tensor<T>? output`, initialized with `null`.
Only that declaration, its using and the suppression change. It must rerun all
sixteen root jobs and prove compiled inference methods, flags and public
declarations still match the measured candidate before performance evidence
can support a release. The source correction alone is not qualification.

[Complete failure census and identities](root-source-policy-failure-20260924.json),
[successor protocol](../observed-dense-where-root-amd-v2/README.md).

Failed closure: `8888b3803c6495dedc87111388e021a4132307bae0dbc490efa6333a5a0a4891`.
