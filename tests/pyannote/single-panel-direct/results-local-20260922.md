# Single-panel convolution input: local proof

The copy-removal component passes all **3,266 cases in both normal and scalar-tail
modes**. An independent check of the accepted `MathOps.PackPanelsB` proves the
layout identity for widths up to 32. This is correctness evidence; AMD timing
and normal product integration remain separate.

Each layout mode checks 1,400 cases / 26,055,764 input values, including 660
identity cases and 296 wider finite counterexamples. Widths are 0–65, 96, 128,
192 and 320; reductions are 0, 1, 9, 63, 64, 65, 288, 576, 1152 and 2304.
Every input, destination and guard bit is checked, including signed zeros,
infinities and NaN payloads. The checker runs with normal hardware support and
with all .NET hardware intrinsics disabled.

The component changes exactly one packing statement in each original consumer.
An admitted candidate with at most 32 columns reads the existing patch as its
packed input; other cases execute the original packing call. All four generated
arithmetic bodies, the scalar-tail reference, every validator and the complete
shape/case manifest remain unchanged. The original production path remains the
comparison control. The normal source wrapper and scratch rental are not yet
changed by this experiment.

## Source findings

`src/Lokad.Onnx/MathOps.cs:PackPanelsB` packs complete 32-column panels, followed
by a row-major tail. At width 32, there is one panel and no rearrangement. Below
32, the entire input is the tail. Width 33 is the first layout requiring a
permutation when the reduction has more than one row. The executed checker
constructs that permutation independently and verifies the boundary.

The closed model census contains **1,968 admitted tiles / 79,204,608 patch values**
with widths at most 32 per embedding graph on each of the three captured crops.
Of those values, 79,200,000 belong to nodes whose full tile width is 32; a final
eight-column tile from the preceding 64-column node adds 4,608. These are static
counts, not measured memory traffic, residency or latency savings.

The earlier ORT source review remains relevant to the convolution/GEMM boundary:
[convolve.cpp](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/convolve.cpp)
forms convolution segments and calls matrix multiplication;
[sgemm.cpp](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/sgemm.cpp)
handles packed matrix panels. Those sources do not establish the installed
wheel's chosen kernel or the performance cost of Lokad's extra copy. This
proposal follows the verified Lokad layout directly.

## Evidence and limits

Windows .NET 10.0.12, CPU2 inherited before CLR, accepted Core `e9c87932`.
Preparation session 53821 exits zero. All 123 resource samples and 17 terminal
process identities pass the independent audit; peak owned RSS is 254,083,072
bytes. Original memory, disk, output and worker-time limits remain intact.

Artifact: `artifacts/pyannote-single-panel-direct-20260922`.
Closure: `77450563eba3ddab07a1c81be4a1921f9c514a700490e1d2cfb90885866afd4f`.
The inherited complete component has closure
`848b9681e1914818a5a9a0a63e98f22b7d1abf06dbc906d243d0527094df2e39`.
The model census has closure
`aaf2a6457d49f087b536b1966427ce1ab580554a568652f79c23a40b85ea4b84`.

The [preceding application trial](../direct-amd-results/results-20260922.md)
remains unselected: its 0.970383 full-request ratio misses the fixed 0.970000
limit. This new component does not reinterpret or retry that unchanged trial.
Production source and accepted ORT benchmark measurements remain unchanged.
