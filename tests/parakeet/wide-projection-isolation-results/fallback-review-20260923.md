# Remaining wide-projection regression

M52 is rejected by the original component gate. All 48 repeatability controls
pass and 20 of 21 case regression checks pass. The twelve original short targets
improve 68.5057%, but fixture 5, 106 rows × reduction 1024 × columns 1024,
rises from 2.753406 to 3.010166 ms (+9.3252%). The 5% bound remains unchanged.
No full-model campaign or product integration follows this result.

The former M50 failure at 167 × 1024 × 4096 now passes: 16.945785 versus
17.192506 ms (+1.4559%). The nine longer cases together improve by 0.4659%.
The all-case sum improves by 43.0303%; it does not override the failed case.

Every clock is retained in six fixed twenty-call blocks. The first three blocks
are warmups; the last three are the original scored interval. These descriptive
blocks change neither the score nor the warmup boundary.

| Process, fixture 5 | 0–19 ms | 20–39 ms | 40–59 ms | 60–79 ms | 80–99 ms | 100–119 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected 0 | 2.814827 | 2.731083 | 2.733906 | 2.737478 | 2.733839 | 2.732201 |
| Candidate 1 | 2.600616 | 2.605442 | 2.595650 | 2.595483 | 3.559307 | 2.807772 |
| Candidate 2 | 2.601652 | 2.627082 | 2.646113 | 2.623027 | 3.260523 | 3.214883 |
| Selected 3 | 2.788878 | 2.746795 | 2.769247 | 2.779566 | 2.778136 | 2.759217 |

Both candidate processes begin this case faster and rise late in the same
interval. This supports a bounded runtime diagnostic before another source
change. It does not identify compilation, collection or another cause, and does
not justify removing those clocks. The separate 80-call codegen processes show
fully optimized copied arithmetic and a 273-byte dispatcher, but cannot identify
which versions executed in these scored calls.

The next diagnostic should reuse the compiled matrix trace consumer and complete
event exporter with the same 21-case, 120-call sequence, ordinary runtime settings
and actual selected/M52 DLLs. Retain every event and raw marker, associate compiler
and suspension intervals, and inspect fixture 5 while preserving every other
case. No unchanged scored retry or retrospective gate change is planned.

[Complete score](screen-20260923.md), [all 504 blocks](screen-blocks-20260923.csv),
[numerical and code qualification](qualification-20260923.md).
The selected release and `BENCHMARK.md` remain unchanged.
