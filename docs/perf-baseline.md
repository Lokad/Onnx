# Perf Baseline (0.2.0 line)

Regenerate: `powershell -NoProfile -ExecutionPolicy Bypass -File bench.ps1`
(Defaults: 1 warmup + 3 timed e5 runs per input, BDN matmul with 2 warmups and 5 iterations.
Raw JSON plus per-run logs land in `artifacts/bench/`, which is git-ignored; this file is the committed snapshot.)

Snapshot below: commit `7b19438`, SDK `10.0.300-preview.0.26177.108`, Release,
Intel i7-14700KF, 31.7 GB RAM, local `models/multilingual-e5-small/model.onnx`.

## e5 single-sample inference (graph exec ms, 1 warmup + 3 timed)

| Input | graphMs | wallMs |
|---|---|---|
| short, 8 tokens | 155, 139, 130 | 1537, 1586, 1526 |
| long, 30 tokens | 268, 253, 268 | 1716, 1632, 1715 |

`graphMs` is the in-process `Executing graph ... completed in` figure (inference only).
`wallMs` is whole-process time including model parse (~230ms), graph build (~90ms),
tokenization, and full-tensor console rendering, so it overstates serving cost.

## MatMul sweep (BenchmarkDotNet mean, InProcess, 2 warmups + 5 iterations)

| Case | scalar | simd | simd intrinsics |
|---|---|---|---|
| 384x384 | 22.205 | 6.183 | 2.606 |
| 384x1536 | 89.005 | 19.293 | 10.882 |
| 6x384x384 | 156.862 | 21.203 | 23.153 |
| 3x4x384x384 | 288.540 | 54.935 | 33.641 |

All figures in ms.

## Notes

- Machine was contended during capture (identical cells vary ~10-20% run to run).
  Compare with same-machine reruns, not across machines.
- The 6x384x384 batched case shows intrinsics slightly slower than plain simd
- Update 2026-09-04: the 6x384x384 inversion was noise. Focused rerun at 15 iterations plus 3 warmups gives scalar 133.0ms, simd 30.3ms (wide CI 22-39ms, high variance), intrinsics 19.0ms. Intrinsics is clearly fastest; the 5-iteration snapshot caught the simd variant on a lucky draw.
  The simd variant variance itself is worth watching in future sweeps; prefer 10+ iterations for batched shapes.
  shape effect; rerun with higher iteration counts before concluding.
- The `me5s-load` / `me5s-run` BDN benchmarks are intentionally excluded: they
  download model and corpus files at benchmark time and are not hermetic.
