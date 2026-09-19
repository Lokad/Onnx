# Exact interleaved BiasGelu experiment — September 19, 2026

Source `0f65c2a1922606e686b8d2d2ef77b11165072e7c` adds an experimental
BiasGelu schedule that interleaves four independent eight-lane erf
computations. Every coefficient and each lane's arithmetic order are preserved.
The isolated AMD kernel improves about 30% on the e5 feed-forward shapes;
the complete-model observations below are smaller and overlap process variation.
The switch remains off by default and outside the ten preferred switches.

Set `LOKAD_ONNX_BIAS_GELU_INTERLEAVED=1` and
`LOKAD_ONNX_BIAS_GELU_INLINE=1` before process startup to enable the route.
It requires AVX-512F, FMA, eight-lane `Vector<float>`, complete bias rows and
a bias length divisible by 32. Other hardware and shapes use existing paths.
The AVX-512 requirement allows additional vector registers; the arithmetic
still uses 256-bit vectors. No approximation or ownership change is introduced.

## Complete-model observations

Both legs use the same frozen core, the ten switches listed in the
[current comparison](comparison-20260919.md), and normal runtime tiering/GC.
The narrow MatMul experiment remains off. The only changed variable is the
new BiasGelu switch. Each visit runs all five cases in the fixed order below;
the visit order is control, interleaved, interleaved, control.

| Tokens | Control mean ms | Interleaved mean ms | Interleaved / control | Early / late control ms | First / second interleaved ms |
|---|---:|---:|---:|---:|---:|
| 8 | 6.0266 | 5.7588 | 0.9556 | 5.9356 / 6.1175 | 5.8112 / 5.7065 |
| 30 | 16.6724 | 16.4081 | 0.9841 | 16.4676 / 16.8773 | 16.4524 / 16.3639 |
| 30 padded to 128 | 65.4140 | 64.3667 | 0.9840 | 65.1436 / 65.6843 | 64.3678 / 64.3657 |
| 128 | 64.9633 | 64.4242 | 0.9917 | 64.3664 / 65.5602 | 63.4310 / 65.4174 |
| 512 | 347.2228 | 338.0271 | 0.9735 | 348.9303 / 345.5154 | 336.2310 / 339.8232 |

All observed aggregate means improve. However, the thirty-token control moves
2.49%, exceeding its 1.59% aggregate improvement; the second 128-token
candidate is slower than the early control. The padded-128 result is more
consistent. These two visits per leg do not establish independent samples,
an uncertainty interval, a certified model speedup or parity with ORT. Do not
combine these ratios with an older ORT timing table to manufacture a new score.

The unchanged schema-4 producer `7c1d2ca` runs five fresh native oracles and
twenty managed workers on CPU 2 of the AMD EPYC 9V74 VM. SDK is 10.0.204,
runtime 10.0.8 and benchmark ORT 1.23.2. Each worker includes 30 seconds of
conditioning, bounded warmup and 33 public Execute measurements. All 660
measured, 27,355 conditioning and 1,108 warmup calls remain. Whole-request
ratios including Reset agree in direction; there are no measured Gen2
collections. Peak sampled RSS is 2,082,672,640 bytes, with maximum observed
foreign CPU fraction 0.001262. These are finite observations.

## Arithmetic and shared-model proof

Twenty-two new tests cover geometry, offsets, guards, in-place use, refusal
without writes, exceptional floats and 2,097,152 random/exceptional values.
Full local suites pass 2,971 backend and 342 tensor tests in both configurations,
with 93 expected ISA skips. Seventy affected tests also pass with hardware
intrinsics disabled. Both settings pass nine native e5 checks and three-mode
determinism; the archive-built core reproduces all six validated mode outputs.

Actual AMD testing passes 282 selected tests per setting, including all 22
new tests, with two expected ISA skips. Every model worker passes complete
native output agreement before and after measurement, maximum scaled error
`1.50055079636e-6`, and input immutability. The existing schema retains native
raw fixtures and full-output error checks; it does not serialize managed raw
e5 tensors.

The same frozen core passes the native DINOv3, ResNet50 and independently
carried GPT-2 regression locally and in both AMD configurations: 106 complete
arrays and 1,286,766 values per replay. All 106 AMD arrays are byte-identical
with the switch off/on. Maximum scaled errors are approximately `1.99e-5`,
`4.30e-6` and `8.95e-6`, respectively. Held outputs, caller inputs and failure/reset
checks pass. This lane uses its separately pinned native ORT 1.29 reference.

Actual product code generation is 1,991 bytes, uses vector registers through
`ymm24`, and contains no calls inside the hot loop. It has five vector stack
stores and five corresponding reads; it is not spill-free. The fallback call
lies outside the loop. The separately compiled prototype was 2,116 bytes,
so its exact instruction layout is not claimed for the product.

## Evidence and disposition

Local ignored evidence lives under `artifacts/gelu-interleaved-product-20260919`
and `artifacts/gelu-interleaved-regression-20260919`. The independent audits
retain every worker, process visit, numerical result, identity and raw shared-model
array. The preceding attribution and arithmetic prototype are under
`artifacts/e5-current-attribution-20260919` and `artifacts/gelu-scheduling-20260919`.

| Artifact | SHA-256 |
|---|---|
| Archive-built core | `c505b6a8d8fece278e82be30a32be89d8e2eeead6598934712a27c192543a463` |
| Source archive | `55802098447c4bd5f46e9340e896791b349bd5807f25966ecda38ec0994cf83d` |
| Product result archive | `8301a51e67a552e348a353e12133a90e0ec4537605e792136de3ec4baf6637fd` |
| Shared-model result archive | `6c47ae2ec8e366af1a4d54ada6daffb9d98d8488a4d9f100723dd334722ee230` |

Retain the narrow opt-in implementation and its evidence. Broader production
configuration qualification is the next priority; repeating width/unroll variants
or unchanged timing calibration is not justified by these observations.
