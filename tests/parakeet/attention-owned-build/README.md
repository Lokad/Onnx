# Qualify the attention owned-preparation policy before model inference

Build the isolated 446-file candidate from attention-owned-source. Retain the
qualified Data binary byte for byte. Compare every compiled Core/Data method,
implementation flag, public declaration and assembly attribute to qualified root
fc116763: only ComputationalGraph.PrepareOwnedMatMulWeights may differ, with no
new or removed product method. Preserve the four existing compiler-warning
occurrences. The arithmetic leaves remain exact.

Run 93 focused contracts in each normal and disabled-AVX512 mode, then 26 in
disabled-intrinsics mode. This includes all unchanged feed-forward preparation
contracts and generic destination/empty/vector contracts, 29 new square-attention
cases and the actual loaded-product/hardware identity check. The new geometry
fact covers every T and 2T-1 from the 19 corpus lengths in both Speed and Memory
execution, including repeat requests, held outputs and zero per-call scratch.
The disabled-intrinsics lane checks preparation refusal and existing logical
fallback behavior. No model inference or application score is part of this lane.

Reuse the bounded worker, CPU 2 for compute/all child threads and CPU 0 for
monitoring. SDK 10.0.204/runtime 10.0.8 and the existing offline feed remain fixed.
Build jobs require 2 GiB available RAM / 1 GiB tmpfs, 3 GiB RSS and 180 seconds;
focused test jobs keep the same memory bounds and allow 300 seconds. Preserve
1 GiB available RAM/tmpfs, 512 MiB output and half-second resource samples.

From repository root, prefix with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/attention-owned-source/prepare.py
    tests/parakeet/attention-owned-build/run.py prepare
    tests/parakeet/attention-owned-build/run.py stage
    tests/parakeet/attention-owned-build/run.py launch build
    tests/parakeet/attention-owned-build/run.py observe build
    tests/parakeet/attention-owned-build/run.py collect build
    tests/parakeet/attention-owned-build/review.py build
    tests/parakeet/attention-owned-build/run.py launch capture
    tests/parakeet/attention-owned-build/run.py observe capture
    tests/parakeet/attention-owned-build/run.py collect capture
    tests/parakeet/attention-owned-build/review.py capture

Freeze source and tools before preparation; every mutating action writes once.
Observe is read-only. Verify terminal owners before collection. Preserve failures
and investigate retained output before recovery; never repeat inference merely
because an audit fails. The root product remains unchanged. The exact model
weight census, complete public results and independent application comparison
remain required after these focused contracts pass.
