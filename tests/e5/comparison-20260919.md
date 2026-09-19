# Current e5 comparison — September 19, 2026

On the AMD EPYC 9V74 VM, the current experimental configuration reduces mean
public inference latency by 38–62% versus the frozen release on the primary
cases. Three primary cases still exceed the target of at most 5% above ORT.
This comparison describes the observed samples; it does not certify parity,
repair failed historical A/A qualification, or enable any default switch.

| Tokens | Release mean ms | Current mean ms | ORT mean ms | Current / ORT | Observed six-triplet ratio range |
|---|---:|---:|---:|---:|---:|
| 8 | 15.3093 | 5.8282 | 6.1527 | 0.9472 | 0.9084–1.0035 |
| 30 | 26.7654 | 16.5017 | 15.2355 | 1.0831 | 1.0445–1.1318 |
| 30 padded to 128 | 110.1884 | 64.8381 | 60.3718 | 1.0740 | 1.0602–1.0878 |
| 128 | 105.0934 | 64.9170 | 60.4942 | 1.0731 | 1.0652–1.0831 |
| 512 | 685.8483 | 340.6492 | 284.7535 | 1.1963 | 1.1797–1.2152 |

The first four cases are primary; 512 is required regression coverage and a
parity stretch target. Ratios divide aggregate means. The range contains the
minimum and maximum of six same-case/round ratios, **not a confidence
interval**. Pooled-median current/ORT ratios are respectively 0.9312, 1.0763,
1.0691, 1.0673 and 1.1964. Reaching 1.05 at the observed ORT means requires
about 0.50 ms less current latency at 30 tokens and 1.4–1.5 ms at 128.

## Configuration and measurement

Release source is `4495fc68b9505b0b6fab73146bd905424588218e`; current source is
`8e93aa72388b404668434aaacfaf7a5ab1e7d42e`. Both use SDK 10.0.204 and runtime
10.0.8 on the same four-vCPU guest, with inference confined to CPU 2. Native
benchmark ORT is 1.23.2. The model is the hash-pinned local multilingual-e5-small
asset documented in the [conformance README](README.md).

Current enables these ten experimental switches before process startup; all
remain off by default in the product:

```text
LOKAD_ONNX_PACKED_AVX512_ROWS=1
LOKAD_ONNX_DEFERRED_RELEASE_CACHE=1
LOKAD_ONNX_RELEASED_BUFFER_CACHE=1
LOKAD_ONNX_FUSED_TEMP_RELEASE=1
LOKAD_ONNX_SOFTMAX_EXP_PRUNE=1
LOKAD_ONNX_BIAS_GELU_INLINE=1
LOKAD_ONNX_VECTOR_TRANSPOSE_FACES=1
LOKAD_ONNX_RELEASE_RESHAPE_VIEWS=1
LOKAD_ONNX_SOFTMAX_EXP_INLINE=1
LOKAD_ONNX_SOFTMAX_NONPOSITIVE=1
```

The newer `LOKAD_ONNX_PACKED_AVX512_NARROW` route is off. Inherited experiment
and runtime variables are cleared; normal tiered compilation and GC apply,
with `DOTNET_TieredCompilation=1` and no forced collection. Release predates
the switches; its explicitly recorded environment is the same.

The prospectively frozen schedule runs every permutation of release, current
and ORT once for each case over six rounds. Each role appears twice at every
position within a triplet. Case order rotates and reverses according to the
fixed schedule. This balances position and predecessor effects; it does not
establish randomization, independent rounds or stationary performance.

The unchanged schema-4 producer at `7c1d2ca` runs five fresh native oracles,
then 90 sequential measured workers. Each worker has 30 seconds of
conditioning, bounded convergence warmup, and 33 measured public Execute/Run
calls. Every process and sample is retained: 2,970 measured calls, 105,860
conditioning calls and 4,180 warmup calls. Six processes and 198 measured calls
contribute to each role/case. No fitted drift correction, filtering or
candidate-dependent stopping is used. See the
[isolated evidence protocol](../../eng/isolated-evidence.md) for boundaries.

Separate complete-request means include Reset/disposal and agree in direction:
current 5.8916 / 16.5652 / 64.9013 / 64.9811 / 340.7136 ms. Load and first-call
costs are retained separately and excluded from warm inference. Current
allocates approximately 0.712 / 0.848 / 1.452 / 1.452 / 3.825 MB per complete
request and has no measured Gen2 collections. Release has 5 / 3 / 23 measured
Gen2 collections in padded-128 / 128 / 512; its expensive calls remain in the
results. Managed allocation counters do not capture all native allocations.

## Validation and retained evidence

An independent audit verifies every worker, schedule order, nonoverlapping
processes, payload/source/runtime identities, confinement, input and oracle
bytes, complete output comparisons, and supervision accounting. All numerical
and input checks pass; maximum scaled output error is `1.50055079636e-6`.
Maximum sampled process RSS is 2,859,339,776 bytes; maximum observed foreign
CPU fraction is 0.00319533196753. Guest exclusivity cannot control hypervisor
neighbors.

Local ignored evidence is under `artifacts/e5-current-comparison-20260919`:
`report.md` includes all process ranges, tails, startup, conditioning and GC;
`summary.json` retains every measured sample and process result. The collected
producer output includes full provenance and supervision. SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| Release core | `7603d8e64b82416822aef347edec7e931e9f800d7d5b4a26078445d4aa1a7cc2` |
| Current core | `07ad08d790972bd69f0e3f99cd3fce5e495a3fd383c5c395d2cf5b92ba6594f4` |
| Result archive | `22d404af8b5ef47f928a7820323aac92998bc99f3e592b275cd291ea5bb97da0` |
| Summary | `43bd8358ae87203df2bf339ee300dccdf38e09b91ab2cb7b64657cc4cbd2350a` |
| Schedule | `7d1d7b048c2a73f623f476cf1f83a80d8649f946e62439a8b036e5c636cad4f4` |

These results rank the remaining work: preserve the substantial existing
gains, investigate the small primary-case residual with complete-model checks,
and keep the larger 512-token gap separate. Statistical timing qualification,
coherent default promotion and audio acceptance remain open.
