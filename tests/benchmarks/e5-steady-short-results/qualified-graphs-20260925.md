# Dispatch relocation: explicitly combined graph qualification

**All eight cases pass.**

| Case | Release (s) | Relocation (s) | Microsoft ORT (s) | Relocation / ORT | Warmups |
| --- | ---: | ---: | ---: | ---: | ---: |
| e5-8tok | 0.005695 | 0.005670 | 0.005604 | 1.011860 | 6000 |
| e5-30tok | 0.016025 | 0.016132 | 0.015423 | 1.045999 | 1200 |
| e5-30pad128 | 0.062422 | 0.062493 | 0.063044 | 0.991270 | 600 |
| e5-128tok | 0.062451 | 0.062564 | 0.062610 | 0.999261 | 600 |
| e5-512tok | 0.335817 | 0.337958 | 0.294278 | 1.148429 | 600 |
| dinov3 | 0.113948 | 0.113883 | 0.101985 | 1.116663 | 600 |
| resnet50 | 0.099765 | 0.100889 | 0.072989 | 1.382244 | 600 |
| gpt2 | 0.034218 | 0.034364 | 0.020187 | 1.702236 | 600 |

Seven cases retain their complete passing observations from def19d3f.
Eight-token e5 uses the separately specified correction above. The original
failed short-e5 row and its campaign remain immutable. This is an explicitly
sourced qualification, not a reinterpretation of the old measurements.

All cases use the same exact product binaries, models, inputs and numerical
bounds. All retain 180 measured calls per timing process. These complete
case observations contain 73,512 calls, 8,640 measurements and 72 setups;
all 78,201 calls across both source campaigns remain retained.

Repeatability: 24/24 controls pass.
Regression: 8/8 gates pass.

[Source protocols and full observations](qualified-graph-observations-20260925.json).
The combined clock index is retained at artifacts/parakeet-owned-batch-graph-qualification-20260925/clocks.csv;
its rows identify their source campaign. No source or BENCHMARK.md promotion
follows: complete Parakeet, shared/Pyannote and package qualification remain.

Original closure: `def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638`.
Correction closure: `b81128a6dbea610cf0571279e078debe3ff24c2b0f3c89bbea0e0c4ab675629a`.
Combined closure: `ec95b9c7f8019b402fe513b6da6938cf83a8bc2c91f1d5ab65fedd4f7b55ed7c`.
