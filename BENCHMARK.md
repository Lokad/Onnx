# Benchmarks

Method, unless stated otherwise: warmed-up runs on the same host, default JIT,
intrinsics enabled (FMA host), single-threaded Lokad (`MaxDegreeOfParallelism`
default 1) against ONNX Runtime CPU EP with its default threadpool. Report
best-of-N and median; treat ~10-20% swings on this contended box as noise.
Rerun: `lonnx benchmark <id>` for op microbenchmarks, `bench.ps1` for the
warmed-up e5 + MatMul sweep, `dotnet run --project tests/Lokad.Onnx.Bench`
for the Lokad-vs-ORT model table below (local models only, nothing downloaded).

## Model level: Lokad vs Microsoft.ML.OnnxRuntime 1.23.2 (refreshed 2026-09-07)

Host LOKAD-0399, FMA. Bench harness: 3 warmup + 7 timed runs per engine,
`graph.Reset()` between Lokad runs, first-output cross-check (all match ORT
to 5 decimals, so the wiring is sound and the gap is pure speed).

| case | Lokad best | Lokad median | ORT best | ORT median | gap (best) |
|---|---|---|---|---|---|
| e5-8tok | 28.5ms | 30.1ms | 2.3ms | 2.5ms | ~12x |
| e5-30tok | 51.9ms | 59.7ms | 3.8ms | 4.1ms | ~14x |
| dinov2-224 | 504.0ms | 510.3ms | 49.4ms | 51.3ms | ~10x |
| dinov3-224 | 318.7ms | 347.8ms | 17.9ms | 19.4ms | ~18x |
| resnet50-224 | 1710.5ms | 1751.5ms | 8.6ms | 10.3ms | ~199x |
| gpt2-4tok | 239.3ms | 286.2ms | 9.0ms | 14.5ms | ~27x |

Two systematic handicaps explain most of it, in order: ORT threads every
MatMul/Softmax across all cores while Lokad runs one thread (R1/R2), and
ORT packs MatMul panels while Lokad streams B from cache (packing assessed,
no win on L2 shapes). The remainder is kernel depth (vectorized exp, MLAS
rational erf) plus per-node framework overhead.

ResNet50 (~199x) is the outlier and the honest Conv story: the Conv kernel
is a direct loop with no im2col/GEMM lowering, no packing and no threading,
against ORT's fully tuned kernels. GPT-2 (~27x) rides the same MatMul and
framework handicaps as the other transformer models. First-output checks
match ORT to 5 decimals on all six cases, so every gap below is pure speed.

Note: CLI `--profile` runs read 2-4x slower than these numbers (e5-8tok
127ms) because per-node profiler stages dominate small graphs; that cost is
instrumentation, not inference, and is R6 material.

## Op microbenchmarks (`lonnx benchmark ops`, 2026-09-05, R15b rows 2026-09-07)

Shapes follow the model profiles (attention batches, 257x384 DINO rows,
201x192 RoPE pairs, 1000x384 embedding table). Modes pinned explicitly.

| op | scalar | simd | intrinsics |
|---|---|---|---|
| Add 257x384 | 140us | 64us | 73us |
| Mul 257x384 | 263us | 61us | 68us |
| Div 257x384 | 259us | 60us | 63us |
| Erf 257x384 | 603us | 124us | 123us |
| Gelu exact 257x384 | 491us | 134us | 124us |
| Softmax 12x30x30 | 51us | - | - |
| Softmax 6x257x257 | 1775us | - | - |
| LayerNorm 257x384 | 767us | - | - |
| Gather 30 rows | 529us | - | - |
| Transpose 12x30x30 | 50us | - | - |
| Concat 2x201x192 | 24us | - | - |
| Conv stem 7x7s2 1x3x112x112 | 12.70ms | - | - |
| Conv 3x3s1 1x64x28x28 | 13.04ms | - | - |
| Gemm 4x768@768x2304+bias | 329us | - | - |
| Tanh 4x3072 (scalar MathF) | 55.5us | - | - |
| Split 4x12x2304 3-way | 33.4us | - | - |
| GlobalAveragePool 1x256x14x14 | 22.4us | - | - |

Reads: Erf/Gelu barely move with SIMD because scalar `exp` dominates (R4
covers both); Softmax on DINO-size attention (1.8ms) wants vectorized exp
(R3); Gather allocates ~2.4MB for a 46KB result, suspicious and queued in
R6; LayerNorm still accumulates in double.

R15b reads (2026-09-07): both Conv shapes cost ~13ms for ~30M MACs (direct loop, no lowering/packing/threading); Gemm allocates 2x its output (MatMul2D temp P plus bias-fold Y, fuse candidate); Tanh is scalar MathF at 55us for 12k elements (ExpVector-based vectorization queued, ~3-4x available); Split/GAP allocate exactly their outputs.
