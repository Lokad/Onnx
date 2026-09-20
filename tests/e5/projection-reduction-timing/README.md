# Complete projection-bank reduction-block timing

This component comparison uses the exact kernels from the closed
[AMD arithmetic/code proof](../projection-reduction-blocks/proof-results-20260920.md).
It adds no product code, model inference or ORT timing. The complete prospective
[ExecPlan](../../../.agent/m2-reduction-block-timing-20260920.md) fixes the workload,
conditioning, sample order, limits and decision rules before execution.

Each of four fresh AMD workers measures five synthetic e5 projection banks.
Every visit clears outputs and computes all72 independent matrices, including
tails. All modes share the same84,934,656 packed weight bytes and activation
layout. Modes0/1 call the identical original;2/3 use reduction blocks128/256.
The padded128 label denotes geometry and a distinct input seed, not model masks.
Warmups require16 cycles and3 compute seconds per mode. Measured cycles use
two locally shuffled complete24-permutation blocks,48 calls per mode/bank.

Full output hashes for every matrix/mode are checked before and after timing;
complete output arrays are not separately retained. Each sample retains raw
stopwatch ticks, allocation and GC counters. Packing and validation are outside
timing. Delegate dispatch, clearing, multiplication and tails are inside.

From the repository root, with .NET10SDK and Python3.13:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/e5/projection-reduction-timing -p test_*.py
    C:/Python313/python.exe -X utf8 -B tests/e5/projection-reduction-timing/prepare.py build --artifact artifacts/e5-reduction-timing-20260920
    # Commit sources before freeze; do not repeat a completed build or launch.
    C:/Python313/python.exe -X utf8 -B tests/e5/projection-reduction-timing/prepare.py freeze --artifact artifacts/e5-reduction-timing-20260920
    C:/Python313/python.exe -X utf8 -B tests/e5/projection-reduction-timing/vm.py launch --artifact artifacts/e5-reduction-timing-20260920
    C:/Python313/python.exe -X utf8 -B tests/e5/projection-reduction-timing/vm.py poll --artifact artifacts/e5-reduction-timing-20260920
    # Only after the original supervisor and all workers are terminal:
    C:/Python313/python.exe -X utf8 -B tests/e5/projection-reduction-timing/vm.py collect --artifact artifacts/e5-reduction-timing-20260920
    C:/Python313/python.exe -X utf8 -B tests/e5/projection-reduction-timing/audit.py --artifact artifacts/e5-reduction-timing-20260920

The local host test supplies no AMD numerical/timing evidence. Workers inherit
CPU2 before CLR startup and run ordinary .NET10.0.8; supervisor uses CPU0.
Per-worker600second/2GiBRSS/1GiBavailable guards retain their telemetry. Guest
CPU/steal counts and boundary foreign-process deltas describe activity; snapshots
can miss short-lived processes and cannot establish hypervisor exclusivity.
Full numerical verification occurs outside measured intervals, not after every
sample. In a forcibly terminated worker, completed-bank JSON and supervisor
telemetry survive; the current bank's in-memory records may not be recoverable.
Such an incomplete execution cannot pass the auditor or supply a verdict.
