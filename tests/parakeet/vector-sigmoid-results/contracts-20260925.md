# Vector Sigmoid: compiled scope and numerical qualification

The isolated candidate changes only `CPUExecutionProvider.Sigmoid` over M78.
It uses the existing portable-vector exponential for float blocks and retains
scalar tails, disabled-SIMD behavior, materialization, ownership and doubles.
No graph fusion, shared exponential coefficient or compiler flag changes.

| Qualification | Result |
|---|---:|
| Normal focused tests | 12 passed, 0 skipped |
| Hardware-disabled focused tests | 12 passed, 0 skipped |
| Sweep values per mode | 2,048,769 |
| Largest normal absolute error against scalar expression | 1.1920929e-7 |
| Largest hardware-disabled error | 0 |
| Existing float tolerance | 1e-6 |
| Unchanged Core methods | 3,276 of 3,277 |
| Unchanged Data methods | 697 of 697 |
| Added warnings | 0 |

Tests cover every vector tail, empty/scalar tensors, special values in every
lane, exponent and rounding boundaries, dense finite and deterministic float
bit-pattern sweeps, reversed/sliced/broadcasted/offset layouts, held-output
independence, doubles, invalid types/options and actual runtime/product identity.
Three retained encoder tests include frozen Microsoft ORT numerical expectations.
Public surfaces, existing method flags and all other compiled bodies match M78.

All six build jobs and both test jobs completed on the AMD VM with
SDK10.0.204/runtime10.0.8. Both test processes used CPU2 and CPU0 monitoring;
26 resource samples pass and peak owned RSS was337,297,408bytes.
No model execution or performance score is claimed. Full native/model validation,
the operator screen and complete application comparison remain pending. The
existing e5-8tok release failure remains unresolved; root and BENCHMARK are unchanged.

Candidate Core:
`9bae182b9c846bd34f5b4f0b4738dcccad60d9aa10a1efd8f08e767dc4d27194`.
Candidate Data (all method bodies unchanged):
`f07b03bce4f319b942221ea352f5097691123141a483cc9d7d02175115bda8bb`.
Source receipt:
`d69b351ecd32b7e3b4aa5cd5595ee4f13d0ed72efd47cf97529cabd522ddf22e`.
Build review:
`11e4d2ecc8963132fdbdb9d059979a87b577696fd745c017cb8158b1070fb1fa`.
Numerical closure:
`0b1929e1df62f1dcae578b9c9a6d229b9a074efa19bd9aa0c740b08d5378a07b`.

Raw evidence: `artifacts/parakeet-vector-sigmoid-build-amd-20260925`.
[Preparation and execution](../vector-sigmoid-build/README.md),
[candidate method](../vector-sigmoid-source/Sigmoid.cs.txt),
[focused tests](../vector-sigmoid-source/SigmoidVectorTests.cs.txt).
