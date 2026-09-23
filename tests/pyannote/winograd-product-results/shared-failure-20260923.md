# Initial Winograd shared-model qualification: retained failure

The original M34 shared campaign failed its requirement that every shared-model
output remain bit-identical. Generic Winograd convolution also applies to
ResNet-50, whose two 2,048-dimensional feature vectors change bits. A structural
ONNX census finds 13 eligible convolutions among its 53 convolutions. DINOv3
has no eligible convolution, and GPT-2 has none; their 104 arrays remain exact.

Both ResNet vectors pass the original native 1e-4 scaled-error limit; the largest
native error is 4.05311584e-6, and the largest change from selected is
1.60932541e-6. All input and held-output checks pass. This does not convert the
frozen exactness failure to a pass. The initial plan incorrectly declared every
shared model unaffected by the generic arithmetic change.

Three children completed with exit code 0: selected shared models, selected e5,
and candidate shared models. The supervisor then rejected candidate exactness
and exited 1. Candidate e5 never launched. All 93 resource observations pass;
peak owned RSS is 2,469,814,272 bytes. Every owner is terminal.

A separate [revised qualification](../winograd-product-shared-amd-v2/README.md)
derives affected models from pinned ONNX structure, retains the original native
limits, and preserves exact comparisons for unaffected models. It independently
audits and reuses these three completed outputs, running only missing candidate
e5. No model-specific product dispatch or application timing is introduced.

Failed closure: `053b003ef278896884f8214a5f11316b63ddf7a922c80688302f5987f9bec902`.
[All retained failure observations](shared-failure-observations-20260923.json).
Full evidence: artifacts/pyannote-winograd-product-shared-amd-20260923.
