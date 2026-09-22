# Single-panel direct-output component on AMD

The new component **passes the original eligibility gates**. All correctness
and resource checks pass. Equal-shape geometric mean candidate / production is
**0.901074**, against the fixed 0.95 limit;
the worst shape is **0.998742**, limit 1.05.
All 44 repeated-process controls pass: **True**.

The candidate combines direct final-output writes with omission of the packing
copy for admitted patches of at most 32 columns. Production retains its ordinary
packed three-row/two-row kernels and bias/copy epilogue. These component timings
include clearing, packing when needed, multiplication and bias/copy. They exclude
patch formation, scratch rental and tensor views. The comparison does not isolate
the causal benefit of copy removal, and supplies no application or ORT ratio.

All 3,266 cases pass in each Windows/AMD normal/scalar-tail mode. The
[independent local layout proof](results-local-20260922.md) additionally passes
1,400 cases per normal/hardware-disabled mode. Arithmetic bodies, validators,
all 22 actual shapes and their output strides remain unchanged.

AMD EPYC 9V74, CPU2, .NET 10.0.8, accepted Core `e9c87932`. Four fresh timing
processes run baseline, candidate, candidate, baseline, retaining six measured
blocks per shape/process after the original conditioning. All 528 raw blocks
and conditioning records are retained in [observations](observations-amd-20260922.json).
No sample exclusion or unchanged timing retry is permitted.

| Rows / reduction / columns | Candidate / production | Baseline max/min | Candidate max/min |
|---|---:|---:|---:|
| 32 / 9 / 1440 | 0.998742 | 1.004004 | 1.006000 |
| 32 / 9 / 1568 | 0.994184 | 1.011679 | 1.006981 |
| 32 / 288 / 160 | 0.920761 | 1.004776 | 1.013100 |
| 32 / 288 / 192 | 0.910345 | 1.002870 | 1.001534 |
| 64 / 32 / 472 | 0.998118 | 1.003997 | 1.012378 |
| 64 / 32 / 672 | 0.995794 | 1.001614 | 1.004168 |
| 64 / 288 / 120 | 0.918614 | 1.023257 | 1.001186 |
| 64 / 288 / 160 | 0.915936 | 1.007941 | 1.007207 |
| 64 / 576 / 88 | 0.943715 | 1.033006 | 1.000935 |
| 64 / 576 / 96 | 0.947880 | 1.005983 | 1.000281 |
| 128 / 64 / 200 | 0.685217 | 1.008103 | 1.003578 |
| 128 / 64 / 320 | 0.670850 | 1.001103 | 1.005227 |
| 128 / 576 / 8 | 0.897333 | 1.035342 | 1.000458 |
| 128 / 576 / 64 | 0.945294 | 1.001468 | 1.000852 |
| 128 / 1152 / 8 | 0.905440 | 1.032230 | 1.001566 |
| 128 / 1152 / 32 | 0.915861 | 1.004059 | 1.000046 |
| 256 / 128 / 130 | 0.830021 | 1.003766 | 1.000967 |
| 256 / 128 / 160 | 0.796045 | 1.003367 | 1.000124 |
| 256 / 1152 / 2 | 0.949593 | 1.003027 | 1.001067 |
| 256 / 1152 / 32 | 0.943072 | 1.028939 | 1.000871 |
| 256 / 2304 / 2 | 0.885635 | 1.014339 | 1.003667 |
| 256 / 2304 / 32 | 0.958274 | 1.002662 | 1.000462 |

All 864 AMD resource samples pass;
peak owned RSS is 275,156,992 bytes.
All seven remote supervisor/target identities are terminal before collection.
The local proof separately retains 123 resource observations and 17 terminal
identities. Original CPU, runtime, memory, tmpfs, artifact and time bounds hold.

The [preceding full application trial](../direct-amd-results/results-20260922.md)
remains unselected at ratio 0.970383 against its 0.970000 gate. This component
does not amend that result. Normal product composition, scratch-accounting
checks, full model/public/meeting qualification and fresh application timing
are required before any root integration. Production remains unchanged.

Artifact: artifacts/pyannote-single-panel-amd-20260922.
Closure: 69a4b633684555b239db996abbbdf435ca7116692bf1cb664011b2c94f4f24df.
Local closure: 77450563eba3ddab07a1c81be4a1921f9c514a700490e1d2cfb90885866afd4f.
Reproduction is vm.py prepare/stage/launch/observe/collect, then audit_vm.py and
this report after terminal owners. Existing output directories are not restart targets.
