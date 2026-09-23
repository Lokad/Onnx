# Winograd numerical feasibility on AMD

This standalone prototype tests F(2x2,3x3) for stride-one convolution. The
four selected direct-kernel source files are copied byte for byte; the product
has no dispatch to the prototype. Its transformed arithmetic may round
differently. Every eligible captured call must retain the original ORT scaled
error bound of `1e-4`; no timing or product benefit is claimed here.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` followed by
`tests/pyannote/winograd-prototype/run.py prepare`, then `stage`, `launch`,
`observe`, and, only after all owners terminate, `collect`. Run `audit.py`
after collection. Do not relaunch a frozen attempt or skip a failed case.

Seven jobs use .NET SDK10.0.204/runtime10.0.8 on AMD CPU2: SDK identity,
restore, Release build, raw256, raw512, captured256, captured512. Raw workers
each execute1,152 cases over eight channel pairs, six spatial shapes, three
patterns and eight bias/residual/ReLU combinations. Each compares the selected
direct and candidate implementations to an independent double-precision direct
convolution. Impulses must be exact. Repeatability, held outputs, read-only
inputs and scratch sentinels are checked. Twenty-one nonfinite/overflow/budget
refusals preserve destination contents. Twenty-four alias and extent contract
checks per raw worker verify rejection before any buffer is modified. Captured workers retain all87
eligible stride-one calls and compare every value against its original native
reference, with unchanged error limits. All numerical failures remain in the
terminal JSON; the independent audit decides numerical admission.

Build preflight10GiB available; numerical preflight12GiB; tmpfs preflight3GiB.
Unchanged live bounds:8GiB owned RSS,1GiB available/tmpfs floors,900seconds/job,
1GiB output and2GiB artifacts. SupervisorCPU0; explicit AVX512 disable only
in the AVX256 correctness workers. Every source, fixture, interpreter and
runtime dependency is pinned before execution. No model or package download.

The transform follows the matrices in
[Lavin and Gray, section4.1](https://arxiv.org/pdf/1509.09308).
Sixteen transformed products yield four outputs, compared with36direct
products per input/output-channel pair. Transforms and memory traffic are
additional costs. ORT's inspected blocked convolution is not claimed to use
this algorithm. The prospective application gate remains at least3%improvement,
with all native, public-result, long-meeting, ownership and regression checks.
