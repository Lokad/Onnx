# Opt-in zero-block softmax: product qualification

Product `087e280` adds `LOKAD_ONNX_SOFTMAX_ZERO_BLOCKS=1`, off by default.
It skips the existing exponential polynomial when every computed argument in
an eight-float vector is below the original zero cutoff. It preserves the
original kernel as fallback and leaves its arithmetic unchanged.

The preceding [long-batch kernel experiment](../softmax-batch-control/results-20260919.md)
qualified the mechanism. This separate protocol checks the actual public graph
API, graph ownership, shared models and complete e5 latency. Its criteria were
fixed before product timing; it does not establish calibrated confidence or
reuse historical ORT latency to claim a new ratio.

The exact tested core is
`187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4`.
These DLLs were built from the implementation source immediately before its
commit. Their metadata therefore predates that commit; they are **not** claimed
to have been built from an extracted source archive.

Local full suites pass 3,027 backend and 342 tensor tests in each setting, with
93 expected backend ISA skips. The native e5 gate passes all three execution
modes, with exact off/on output bytes. DINOv3, ResNet50 and GPT-2 pass 106 complete
arrays in each setting, also byte-identically. Fallback combinations cover
nonpositive exponentiation off, wider exponentiation on, reciprocal on and
hardware intrinsics off.

The five AMD qualification workers have completed and passed independent
collection/audit: 280 contract tests per setting, 106 shared arrays per setting,
and one separate padded128 codegen worker. The actual 3,744-byte full Tier1
kernel branches around all three polynomial instances; see
[the code inspection record](codegen-review-20260919.json). Codegen timings are
excluded from performance evidence. The complete-model comparison follows
only after this qualification receipt is accepted by the supervisor.

## Frozen complete-model protocol

AMD EPYC 9V74, .NET 10.0.8, CPU 2 inherited before CLR startup, public
`ExecutionOptions.Memory`, normal runtime tiering and GC. Ninety sequential
fresh processes cover five cases, six visits each, and three roles per visit.
`controlA` and `controlB` have identical settings; only `candidate` sets the
new switch. Every case sees all six role orders, with rotating/reversing case
order. `evaluate.schedule()` independently reconstructs the schedule.

Each process retains loading and first-call times, conditions for thirty
cumulative Execute seconds, then records 33 Execute calls and a separate 33
Reset-plus-Execute calls. All 5,940 measured calls, all conditioning calls,
allocation, GC, memory, complete native-reference outputs, input preservation
and held-output checks remain. There are no timing exclusions, forced GC,
CPU-time substitutions or profiler instrumentation in these workers.

For **both** timing boundaries, controlB/controlA must remain within 1% in
aggregate and 3% in every triplet. Candidate divided by the arithmetic mean of
both controls must improve padded128 by at least 1% in aggregate, with no
triplet above 1.02. Other cases may regress at most 2% in aggregate and 5% per
triplet. Foreign CPU may use at most 2% of machine capacity and steal at most
0.5%. Every numerical, identity, ownership and resource check is mandatory.
These are empirical mechanism criteria, not root-program timing calibration
or automatic permission to change defaults. Failed controls make the
performance conclusion inconclusive.

## Tools and evidence

`run.py`, `launch.py` and `collect.py` are the frozen two-phase AMD tools.
`model/Program.cs` is the public graph runner; `shared/Program.cs` is the
complete shared-model replay. Build projects reference explicitly frozen local
dependencies. Rebuilding creates new binary identities and requires fresh
qualification; never replace a frozen DLL under an old receipt.

The local artifact is `artifacts/softmax-zero-product-20260919`. `payload` holds
the immutable 151-file deployment inventory; `provenance.json` pins 187 existing
model/reference assets, the schedule and criteria. Model weights are not
copied. The VM checkout remains `172181fc5ab4eb2bdc2eb7f37e80d25e482a0887`.

Run the independent tools from the repository root:

    python -m unittest discover -s tests/e5/softmax-zero-product -p test_*.py
    python tests/e5/softmax-zero-product/audit.py --artifact artifacts/softmax-zero-product-20260919 --phase qual --output <new-audit.json>
    python tests/e5/softmax-zero-product/audit.py --artifact artifacts/softmax-zero-product-20260919 --phase model --output <new-audit.json>

The model phase requires a successful qualification audit bound to the exact
bundle and terminal supervisor identity. Observe a running process by its
existing PID/start identity; an observation timeout is not a reason to start
another worker. Collect only after the supervisor and every owned process have
terminated. The collector and auditors refuse existing outputs. NumPy is used
only by the offline auditor to compare every exported float with pinned native
arrays. These native values supply correctness checks, not new ORT timing.
