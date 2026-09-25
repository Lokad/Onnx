# Dispatch relocation: short-e5 repeatability failure

The complete graph campaign is **not admitted**. All numerical checks and
all eight regression gates pass, but only 23 of 24 repeatability controls pass.
The candidate's two e5-8tok means are **10.549127 and 12.038724 ms**, ratio
**1.141206**, above the unchanged 1.10 limit. Its 0.997092 mean ratio to
release is therefore not a qualified improvement or proof the old failure is fixed.

This analysis partitions every retained short-e5 call into consecutive blocks
of 30. All 4,680 calls appear; no block, warmup or measurement is selected out.
The tables are diagnostic descriptions, not replacement scores.

| Calls, zero-based | Phase | Release A ms | Candidate A ms | ORT A ms | ORT B ms | Candidate B ms | Release B ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–29 | warmup | 14.420508 | 14.759726 | 5.904999 | 6.132197 | 15.046198 | 14.549348 |
| 30–59 | warmup | 9.494884 | 9.667069 | 5.934300 | 5.994821 | 9.864574 | 9.573753 |
| 60–89 | warmup | 9.548677 | 9.718822 | 5.955025 | 5.984541 | 9.831376 | 9.563292 |
| 90–119 | warmup | 9.423251 | 9.840755 | 5.919628 | 5.932383 | 9.910477 | 9.530167 |
| 120–149 | warmup | 10.300457 | 12.381009 | 5.919922 | 5.934645 | 10.720206 | 11.341724 |
| 150–179 | warmup | 10.249914 | 11.714894 | 5.971465 | 5.945316 | 10.417141 | 11.206708 |
| 180–209 | warmup | 10.320260 | 12.432737 | 5.913167 | 5.996885 | 10.479802 | 11.257049 |
| 210–239 | warmup | 12.356651 | 12.226814 | 5.914282 | 5.971170 | 12.964319 | 12.775061 |
| 240–269 | warmup | 12.238140 | 12.113757 | 5.909621 | 6.040173 | 12.034599 | 11.880031 |
| 270–299 | warmup | 12.189071 | 12.374267 | 5.903610 | 5.966065 | 12.366560 | 11.999192 |
| 300–329 | warmup | 12.439397 | 12.107746 | 5.860290 | 5.976088 | 12.209806 | 11.781159 |
| 330–359 | warmup | 12.154437 | 12.142878 | 5.953200 | 6.026151 | 12.115833 | 11.793673 |
| 360–389 | warmup | 12.348278 | 12.514658 | 5.952372 | 6.042993 | 12.600202 | 12.200076 |
| 390–419 | warmup | 12.206905 | 12.409525 | 6.018363 | 5.991598 | 12.415817 | 12.148769 |
| 420–449 | warmup | 12.250521 | 12.364078 | 5.997247 | 5.952832 | 12.511450 | 12.128650 |
| 450–479 | warmup | 14.574672 | 14.977742 | 6.085316 | 5.967069 | 12.524827 | 15.050403 |
| 480–509 | warmup | 11.626847 | 10.255081 | 6.024254 | 5.958473 | 12.280863 | 10.428154 |
| 510–539 | warmup | 10.905046 | 10.283744 | 5.958391 | 5.971126 | 12.410422 | 10.436320 |
| 540–569 | warmup | 10.869783 | 10.146958 | 5.977125 | 5.957321 | 15.404330 | 10.467762 |
| 570–599 | warmup | 10.891531 | 10.132182 | 5.954200 | 5.957738 | 11.865047 | 10.465260 |
| 600–629 | measured | 10.868413 | 10.216554 | 5.982819 | 5.976284 | 11.880026 | 10.451132 |
| 630–659 | measured | 10.884032 | 10.092290 | 5.967941 | 5.953868 | 11.906973 | 10.513663 |
| 660–689 | measured | 11.624743 | 10.929197 | 5.997940 | 5.968142 | 11.935881 | 11.420704 |
| 690–719 | measured | 10.811648 | 10.104908 | 5.984305 | 5.946587 | 12.015741 | 10.558573 |
| 720–749 | measured | 10.815995 | 10.158974 | 6.003958 | 5.945405 | 12.531806 | 10.450849 |
| 750–779 | measured | 14.214061 | 11.792836 | 5.981585 | 5.947522 | 11.961918 | 13.308613 |

The difference spans several measured blocks: Candidate A is mostly near
10.1 ms, Candidate B near 11.9 ms, while both ORT processes remain near 6.0 ms.
Release and Candidate A also have slower blocks. These clocks do not identify
a particular compilation, collection, CPU-frequency or native-code event.

The earlier [exact-product tier observation](../../benchmarks/e5-direct-tier-results/report-20260925.md)
already showed release f95a13c5 and parent 40260aef receiving final matrix
tiers 13.910–15.148 seconds after the first call, beyond the original prefix.
Those are different, instrumented processes and the candidate there is not
e07a4518. They motivate checking the short benchmark's runtime phase; they
cannot assign a tier or historical cause to the present uninstrumented calls.

Restoring the shared dispatcher's compiled instructions did not establish
repeatable short-e5 performance. Stop this relocation's release path; the
conditional Parakeet model stage has not been prepared or executed. Preserve
this verdict and the parent M78 failure. No unchanged scored retry, nearby
product variant, threshold change or retroactive warmup change follows.

The next question is whether the scored short-call prefix measures a stable
runtime phase for the exact current products. Inspect retained compilation
evidence before proposing a bounded observation or prospective measurement
correction. Any such correction requires its own evidence and cannot relabel
this failed campaign as passed. No new optimization is selected.

[Full verdict and all clocks](graphs-20260925.md),
[exact block fractions](repeatability-blocks-20260925.csv),
[bound evidence](repeatability-20260925.json).

Closure: `def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638`.
