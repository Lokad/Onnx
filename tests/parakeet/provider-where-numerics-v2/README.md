# Both-boundary provider Where qualification

Prerequisite:provider-only buildae46692f,source64460d04. Generic Tensor.Where
and3188otherexistingCore/697Data methods/flags/API remain selected. This lane
qualifies actual selectedCore672e5f30 and candidateCored8a8d8eb through both
Tensor<T>.Where and CPUExecutionProvider.Where. No performance timers exist.

The fixed203-case census includes the original131cases,15boundary cases at
4095/4096/4097elements across false/true/first/last/alternating masks,two large
nonscalar-x cases,eight large raw-Boolean cases,24large layout cases,six large
rank fallbacks,six large invalid broadcasts,and11large scalar float-bit cases.
Fifteen cases intentionally fail:12broadcast exceptions andthree null inputs.
The tensor boundary retains NullReferenceException for nulls; the provider must
return its exact missing-input OpResult instead. The188valid cases return the
coordinate oracle's exact bits and own their storage, with immutable full input
stores/guards, changed-input exactness and preserved held outputs. All prior131
reference output/mutated-output hashes remain pinned by the independent audit.

The unchanged uniform helper admits73cases. The provider's large/scalar guard
attempts91 and admits35; the original122valid timing cases still contain12large
uniform cases. Reflective helper checks are untimed numerical assertions only.
Every numerical provider invocation runs in an isolated profiler node; first-call stage
sequences must include GraphOrchestration, Where validation, and three pairs
of BroadcastTo validation/CalculateIndices: four validation stages in total.
A successful helper has one; an attempted-but-refused helper has five.
V1 omitted these nested calls and stopped in the selected release; its failure
is retained at9bc68920. V2 records and checks every stage in order.
No stage durations are treated as comparative timings.

Each numerical worker additionally checks20provider contracts:eight valid
option modes(default/null,scalar,SIMD,intrinsics,memory,parallel2,disabledpool),
three missing inputs,wrong condition/mismatched operands,three unsupported
integer types,two invalid option combinations,andtwo validation-order cases.
Success and failure OpResult metadata, messages, ownership and input immutability
must match the selected product. Invalid options cannot be bypassed by fast success.

Run current/candidate numerical workers in both ordinary and AVX512-disabled
modes. Two separate ordinary code-generation workers, with profiling disabled, exerciseall188valid cases
through both boundaries81times each, retaining complete provider/tensor/helper/
consumer assembly. The latter are diagnostics, not a screen or proof of future
timed tiers. Inspect full bodies after numerical closure.

One identical consumer build serves both products. Nine serial jobs,CPU2compute/
CPU0monitor,SDK10.0.204/runtime10.0.8,offlinefeed,no Windows build/inference.
12GiBavailable/3GiBtmpfs preflight,8GiBRSS,1GiBremainingmemory/tmpfs,900s/job,
fourhours/campaign,1GiBoutput/job,2GiBcampaign. Original source/closures/assets/
runtime identities are pinned;32captured fixture files are hardlinked after
verification. No model downloads or root product changes.

Freeze tools before C:/Python313/python.exe -X utf8 -B with
tests/parakeet/provider-where-numerics-v2/run.py prepare,stage,launch,observe,collect,
then audit.py. Refuse existing namespaces; collect only terminal PID/birth owners.
Local:artifacts/parakeet-provider-where-numerics-amd-v2-20260924.
VM:/dev/shm/lokad-parakeet-provider-where-numerics-v2-20260924.
Admission requires every case, exact oracle/hash, result/error invariant and
resource check. It permits generated-code review and then a separately frozen
complete-provider performance screen; it cannot update BENCHMARK.md.
