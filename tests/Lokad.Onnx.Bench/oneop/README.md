# One-op and region fixtures

Each directory holds a frozen `model.onnx` built by `generate_oneop.py`
(single shared seed stream; appending sections never changes earlier draws).
The bench runner (`OneOpBenchmarks.cs`) loads every fixture into a Lokad graph
and a single-CPU ORT ALL session on identical seeded inputs and requires
1e-4 scaled agreement before timing.

Buffer reuse contract (E04): both legs reuse their input tensors and their
prepared session/graph across all timed iterations; each timed call allocates
only its output vector, which both engines copy out the same way. Lokad legs
run warmed public Execute, ORT legs warmed session Run, with in-process engine
order rotation. No leg reallocates inputs per iteration.

| Fixture | Mirrors | Constants vs runtime | Layout / kernel branch |
|---|---|---|---|
| matmul_384 | GEMM tile probe | const B | row-major packed panels |
| softmax_attn | E5 12x30x30 scores softmax | runtime | span softmax default |
| layernorm_384 | norm probe 257x384 | runtime | vector norm |
| gelu_384 | exact GELU probe | runtime | exact GELU span |
| conv_3x3 | pointwise probe 64ch | const filters | im2col plus GEMM |
| gemm_1024x256 / gemm_512x4608 | conv-derived GEMM tiles | const B/C | packed panels (512x4608 weights scaled +-0.125 for gate headroom, timing-neutral) |
| conv_3x3_512 / conv_1x1_1024 | ResNet50 layer3/4 pointwise and spatial | const filters | im2col plus GEMM |
| matmul_30x384x1536 / matmul_30x1536x384 / matmul_30x384x384 | E5 30tok MLP up/down and QKV tiles | const B | packed panels |
| gelu_30x1536 / transpose_30x12x32 | E5 MLP activation and QKV shuffle | runtime | exact GELU / tiled transpose |
| gemm_4x768x2304 / gemm_4x3072x768 | GPT-2 prefill c_attn/c_proj tiles | const B plus bias | packed panels |
| matmul_1x201x384x1536 / matmul_1x201x1536x384 / matmul_1x201x384x384 | DINOv3 201tok MLP/QKV tiles | const B | packed panels |
| softmax_6x201x201 | DINOv3 6x201x201 scores softmax | runtime | span softmax default |
| gelu_1x201x1536 / transpose_1x201x6x64 | DINOv3 activation and shuffle | runtime | exact GELU / tiled transpose |
| matmul_1x128/512 tiles | E5 long-regime MLP tiles | const B | packed panels |
| attn_1x6x201x64 / attn_1x6x201x201 / attn_1x12x30x32 / attn_1x12x30x30 | attention scores/context tiles, constant-B side | const B | unpacked activation side |
| matmul_1x8 tiles | E5 8tok MLP tiles | const B | packed panels |
| attnblock_e5_30 / attnblock_e5_8 | E5 attention block QKV/transpose/scores/Div-scale/softmax/context/merge/outproj | const proj weights plus Div scale; runtime activations | transposes plus span softmax plus packed outproj |
| resblock_rn50 | ResNet bottleneck proxy 128ch stride-1 (downscaled layer2.1 pattern) | const filters | im2col plus GEMM plus fused ConvRelu/AddRelu epilogues |
| conv_7x7_stem / conv_1x1_s2_2048 / conv_3x3_s2_512 / conv_3x3_512_7 / conv_3x3_64_56 / conv_3x3_128_28 / conv_3x3_256_14 | ResNet stem, stride-2, and per-layer spatial singles | const filters | im2col plus GEMM |
| mlpbias_e5_30 / mlpbias_e5_8 | E5 MLP up/bias/exact-GELU region | const proj plus bias | BiasGelu fusion path |
| chain_softmax/matmul/transpose 1/2/4/8 | node-count slopes for dispatch sizing | mixed | session overhead intercept |
| attnmask_e5_30 (E04) | GPT-2 scale/transpose/mask region at E5-30 shape: two-live MatMul, runtime Mul scale, causal mask Add, softmax, context | runtime q/kt/v/scale/mask, zero constants | unpacked activation MatMuls plus span softmax |
| attnblock_dino_201 (E04) | DINOv3 S=201/H=6 attention block, Div-scale, no mask | const proj weights (scaled +-0.125 for gate headroom, timing-neutral) plus Div scale; runtime activations | transposes plus span softmax plus packed outproj |
| resblock_s2_trans (E04) | ResNet layer2.0 transition bottleneck at true widths with 1x1-s2 shortcut | const filters (scaled +-0.25 for gate headroom, timing-neutral) | im2col plus GEMM plus fused epilogues |
| matmul_runtime_ab (E04) | activation-by-activation scores MatMul, zero constants | fully runtime | unpacked both sides |

Weight-scaling note: a few region fixtures scale random weights down with the
same draws (timing-neutral) because raw +-1 weights make deep regions amplify
fp-ordering noise about 100x (measured per-prefix: 6e-06 projection diffs grow
to 1.35e-02 at the dino block output), drowning the 1e-4 tripwire without any
wrong computation. Scaled draws keep full shapes and op sequences while
agreeing with margin (dino 3.0e-05, s2trans 1.1e-05).