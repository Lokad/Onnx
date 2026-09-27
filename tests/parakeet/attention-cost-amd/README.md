# Measure current attention packing, multiplication and final-row costs

Route review e83b34fb binds the current source to all 120 attention weights and
their fresh complete-call clocks. The 92 weights without retained preparation
account for 1.091125 seconds of the 1.424700-second attention difference to ORT.
Those totals cannot distinguish packing from arithmetic. This observer answers
only that unresolved question and selects no optimization.

Snapshot qualified source c28b3848 (root closure fc116763, 445 source inputs).
Add reversible hooks in CPUExecutionProvider.MatMul and
Tensor<T>.RunIsolatedShortWidePackedRows plus one private helper. Observe all
120 square weights exactly once per request to expose omitted or unexpected
calls. Record input layout, weight identity, actual preparation lookup and entry
into the fallback; time disjoint scratch rent, packing, multiplication, scratch
return and final-row intervals. Preserve each interval and the unassigned
complete-call remainder. Prepared accepted calls have no observed arithmetic
leaf; never label that absence as a fresh assembly observation.

Keep arithmetic leaves, Data and the original SampledAudio consumer unchanged.
Before inference, compare all 3,288 original Core and 697 Data methods, flags,
public declarations and assembly attributes. Exactly two original Core bodies
may differ, and all additions must belong to the private probe. Existing compiler
warnings must remain the same. No Windows .NET builds or model downloads.

One unchanged control precedes one observed process. Each runs the existing
20 warmups and three measured passes over all 20 clips: 160 complete requests
overall and 9,600 observed attention calls. Require exact complete public results,
input identities and ownership checks. Join every observed call to its request,
weight, current route binding and exact frame count. Validate all interval
boundaries and keep all raw clocks. An observed/control corpus ratio above 1.05
makes stage times unusable for choosing a candidate. Never subtract overhead or
publish these diagnostic timings in BENCHMARK.md.

Reuse the bounded worker with the current inspector's required fourth argument
(its unchanged dependency directory). Compute and child threads stay on CPU 2, monitor
on CPU 0, SDK 10.0.204 and runtime 10.0.8. Builds require 2 GiB available RAM /
1 GiB tmpfs, with 3 GiB RSS and 180 seconds per job. Captures require 11 GiB RAM /
2 GiB tmpfs, with 12 GiB RSS and 900 seconds per job. Preserve the 1 GiB free
RAM/tmpfs floor, 512 MiB output ceiling, half-second resource samples and 1%
foreign-CPU bound. Verify terminal owners before collecting or starting work.

From repository root, prefix with C:/Python313/python.exe -X utf8 -B:

    -m unittest discover -s tests/parakeet/attention-cost-amd -p test_*.py
    tests/parakeet/attention-cost-amd/run.py prepare
    tests/parakeet/attention-cost-amd/run.py stage
    tests/parakeet/attention-cost-amd/run.py launch build
    tests/parakeet/attention-cost-amd/run.py observe build
    tests/parakeet/attention-cost-amd/run.py collect build
    tests/parakeet/attention-cost-amd/audit.py build
    tests/parakeet/attention-cost-amd/run.py launch capture
    tests/parakeet/attention-cost-amd/run.py observe capture
    tests/parakeet/attention-cost-amd/run.py collect capture
    tests/parakeet/attention-cost-amd/audit.py capture

Freeze tools before prepare. Prepare, stage, launch, collection and audit write
once; observing status is read-only. Preserve every failure and investigate
retained output before recovery. No failure authorizes repeating inference.
