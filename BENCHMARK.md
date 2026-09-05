# Benchmarks

Method, unless stated otherwise: warmed-up runs on the same host, default JIT,
intrinsics enabled (FMA host), single-threaded Lokad (`MaxDegreeOfParallelism`
default 1) against ONNX Runtime CPU EP with its default threadpool. Report
best-of-N and median; treat ~10-20% swings on this contended box as noise.
Rerun: `lonnx benchmark <id>` for op microbenchmarks, `bench.ps1` for the
warmed-up e5 + MatMul sweep, `dotnet run --project tests/Lokad.Onnx.Bench`
for the Lokad-vs-ORT model table below (local models only, nothing downloaded).

## Model level: Lokad vs Microsoft.ML.OnnxRuntime 1.23.2 (2026-09-05)

Host LOKAD-0399, FMA. Bench harness: 3 warmup + 7 timed runs per engine,
`graph.Reset()` between Lokad runs, first-output cross-check (all match ORT
to 5 decimals, so the wiring is sound and the gap is pure speed).

| case | Lokad best | Lokad median | ORT best | ORT median | gap (best) |
|---|---|---|---|---|---|
| e5-8tok | 35.0ms | 35.3ms | 2.4ms | 2.7ms | ~14x |
| e5-30tok | 66.1ms | 68.7ms | 3.9ms | 4.0ms | ~17x |
| dinov2-224 | 541.7ms | 570.2ms | 46.5ms | 47.5ms | ~12x |
| dinov3-224 | 432.9ms | 443.3ms | 16.3ms | 19.3ms | ~26x |

Two systematic handicaps explain most of it, in order: ORT threads every
MatMul/Softmax across all cores while Lokad runs one thread (R1/R2), and
ORT packs MatMul panels while Lokad streams B from cache (packing assessed,
no win on L2 shapes). The remainder is kernel depth (vectorized exp, MLAS
rational erf) plus per-node framework overhead.

Note: CLI `--profile` runs read 2-4x slower than these numbers (e5-8tok
127ms) because per-node profiler stages dominate small graphs; that cost is
instrumentation, not inference, and is R6 material.

## Op microbenchmarks (`lonnx benchmark ops`, 2026-09-05)

Shapes follow the model profiles (attention batches, 257x384 DINO rows,
201x192 RoPE pairs, 1000x384 embedding table). Modes pinned explicitly.

| op | scalar | simd | intrinsics |
|---|---|---|---|
| Add 257x384 | 140us | 64us | 73us |
| Mul 257x384 | 263us | 61us | 68us |
| Div 257x384 | 259us | 60us | 63us |
| Erf 257x384 | 631us | 410us | 416us |
| Gelu exact 257x384 | 622us | 432us | 441us |
| Softmax 12x30x30 | 51us | - | - |
| Softmax 6x257x257 | 1775us | - | - |
| LayerNorm 257x384 | 767us | - | - |
| Gather 30 rows | 529us | - | - |
| Transpose 12x30x30 | 50us | - | - |
| Concat 2x201x192 | 24us | - | - |

Reads: Erf/Gelu barely move with SIMD because scalar `exp` dominates (R4
covers both); Softmax on DINO-size attention (1.8ms) wants vectorized exp
(R3); Gather allocates ~2.4MB for a 46KB result, suspicious and queued in
R6; LayerNorm still accumulates in double.
