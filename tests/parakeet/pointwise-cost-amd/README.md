# Attribute current Parakeet pointwise costs

This diagnostic changes three Core method bodies in an isolated snapshot of
qualified source edc1a6c8. Exact inverse source edits restore the original files.
The original arithmetic leaf, Data assembly, application consumer, models and
outputs remain unchanged. No optimization candidate is selected or measured.

Observe every selected convolution with batch one, 1,024 input channels,
1,024/2,048 output channels, group one and size-one kernel. Validate stride,
padding and bias rather than silently dropping an unexpected selected case.
Record output initialization, materialization, the matrix destination clear,
packing and the actual two-row arithmetic leaf. Keep total time and its remainder,
input layouts, object reuse, physical scratch length and each stage's count.
Join every one of 3,840 calls to the unchanged consumer's 80 request intervals;
require 48 calls in the correct alternating filter order for every clip/pass.
Only the 60 measured requests contribute diagnostic corpus totals.

The unchanged control runs first, then the observed product, each with the
existing 20 warmups and three measured passes over all 20 clips. All public
records, input hashes, ownership checks and complete results must match the
retained actual-root control. Require observed/control corpus time <=1.05 to use
the split for selecting an intervention; retain failures and every raw clock.
Do not subtract observer overhead or promote these times to BENCHMARK.md.

The original bounded worker keeps compute and all child threads on CPU 2 and
monitoring on CPU 0. Builds require 2 GiB available RAM / 1 GiB tmpfs, with a
3 GiB RSS and 180-second limit per job. Each capture requires 11 GiB available
RAM / 2 GiB tmpfs, with a 12 GiB RSS and 900-second limit. Preserve at least
1 GiB RAM/tmpfs and the 512 MiB output bound, half-second resource observations
and the original 1% foreign-CPU bound. Use SDK 10.0.204/runtime 10.0.8.

From the repository root, prefix commands with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/pointwise-cost-amd/run.py prepare
    tests/parakeet/pointwise-cost-amd/run.py stage
    tests/parakeet/pointwise-cost-amd/run.py launch build
    tests/parakeet/pointwise-cost-amd/run.py observe build
    tests/parakeet/pointwise-cost-amd/run.py collect build
    tests/parakeet/pointwise-cost-amd/audit.py build
    tests/parakeet/pointwise-cost-amd/run.py launch capture
    tests/parakeet/pointwise-cost-amd/run.py observe capture
    tests/parakeet/pointwise-cost-amd/run.py collect capture
    tests/parakeet/pointwise-cost-amd/audit.py capture

Prepare, stage, launch, collect and publish once. Confirm each owner is terminal
before collection. Reconcile compilation against all 3,286 original Core and 697
Data methods before capture. Only the three named methods may differ; additions
must belong to the private observer. Public declarations, assembly attributes
and original implementation flags must match. An audit failure never authorizes
repeating the workload. Diagnose retained output first.
