# Parakeet sigmoid: the retained instruction addresses select one test

The completed offline review supports testing the ORT-shaped AVX-512 rational
loop. It does not establish a performance improvement. The qualified release
remains 43.459339 seconds versus ORT 39.202810 seconds, ratio 1.108577.

All 115,795 periodic samples and 120 request boundaries join exactly to the
original 676,767-event stream. No new inference was run. The reader lost no
events or stacks. Method rundown ranges, captured JIT bytes and instruction
addresses identify the optimized Tier1 code used by the samples.

Within the sixty measured requests on the application thread, 2,178 leaf samples
land in `SigmoidRationalVector` and 35 in the public `Sigmoid`: 98.4184% of these
2,213 samples are in the arithmetic helper. Every request has helper samples.
Every helper address is in the vector loop; none is in its scalar exponential
tail. Of the 35 public leaf samples, 32 land immediately after the array
allocation helper call. These counts are periodic samples, not measured cycles
or a per-node allocation-time estimate.

The earlier thread-time export gave the public operator much greater weight.
Microsoft's pinned TraceEvent 3.1.23 implementation feeds both allocation events
and periodic samples into its thread-time computer, assigning elapsed intervals
to stacks. Its weighted percentages therefore cannot be interpreted as periodic
instruction-sample fractions. See the [official implementation](https://github.com/microsoft/perfview/blob/v3.1.23/src/TraceEvent/Computers/SampleProfilerThreadTimeComputer.cs),
especially the event subscriptions and `AddCPUSample`. The earlier evidence is
retained; this review resolves that specific ambiguity.

The observed managed loop processes eight floats with 256-bit vectors, nine
fused multiply-adds, a division and five `vfixupimmps` instructions. The previously
proved ORT `MlasSiluKernelAvx512F` processes two sixteen-float vectors per main
iteration using the same rational coefficients. Its native min/max operations
are followed by explicit NaN preservation. That native proof is already bound
to the exact binary, settings and complete nonclock request results of the fresh
profile; another native capture is unnecessary.

Test one fixed implementation of that AVX-512 rational strategy in an isolated
source snapshot. Preserve coefficients and their evaluation order, output
allocation, graph structure and the separate multiply. Require intrinsics and
AVX-512 support; use the existing portable helper for the remainder and fallback.
No vector-width, unroll or coefficient sweep is justified. Check generated code
and special-value, tail, ownership and fallback contracts before application
scoring. The prospective target is at least 0.45 seconds saved per complete
twenty-clip corpus, approximately 1.04% of current latency. This is a falsifiable
target, not an estimate obtained by subtracting unrelated diagnostic clocks.

Fresh node attribution reports 0.944548 seconds across all 96 encoder sigmoid
operators. The 72 SiLU pairs account for 0.852870 seconds of sigmoid and 0.115477
seconds of separate multiply, versus 0.174952 seconds for ORT QuickGelu. The
remaining 24 gate sigmoids take 0.091678 seconds versus 0.020997 seconds. Periodic
samples are joined to requests, not individual nodes; do not assign them only
to the 72 SiLU sites. The sample capture had 13.52% observation overhead and
native callees remain partly opaque. Individual hot instruction offsets do not
prove which instruction limits throughput.

Admission still requires the original six-process current/candidate/ORT/ORT/
candidate/current application comparison: all 63 repeatability controls and
21 gates, at least 1% matched corpus improvement and no clip regression over 5%.
Broader model and actual-package qualification precedes promotion. Parity at
1.05 or better remains a separate goal.

Evidence: `artifacts/parakeet-sigmoid-address-review-20260928/closed.json`,
SHA256 `a0ae70ee6e91581ad9f606c3862c8dadef3ff251b6e4218c8beceb7ecfee1c1e`;
analysis `744e9985012a8f1d7b36eda129043a1d18e30b218946653bf8c42237145f4497`.
The original capture closure is `31b91fba1f513c219a7f1c4b9c44b637ff90389d676522856b64be83c5280a54`.
The native routine proof is `41f851c4e473856383c6d51f7b5a0e86d5f0ff8c47947df1f4176b35e2301fc3`.
Offline supervisor 1292231 and PowerShell 1292232 both exited zero; fifteen files
were collected once. The 106,274,653-byte derived ETLX remains on the VM.
