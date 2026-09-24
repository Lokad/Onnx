# Parakeet complete-provider Where screen

This prospective screen compares selected Core672e5f30 with provider candidate
Cored8a8d8eb on the exclusive AMD VM. No product source is changed. Numerical
V2 closure and complete code review are prerequisites, independently pinned
alongside source64460d04, buildae46692f and selected release16d57081.

Use all122 valid cases from the qualified131-case census, in its existing order.
The nine intentional error cases remain covered by numerical qualification.
The fixed partitions retain six actual captured float targets,42other uniform
cases,74remaining cases. Only12of122cases meet the provider specialization
guard; smaller uniform cases execute the original Tensor.Where. Partitions
remain identical to the prospective M57 plan; no case is removed. These include the two actual Int64 calls,18large mixed
captured masks, custom tensors, reversed tensors, strided views, high ranks,
aliasing, raw Boolean bytes, other dtypes, scalar outputs and nonscalar x.
No case is selected using timing. The qualified coordinate oracle and exact
input construction are retained; preparation verifies their source identity.

Four fresh ordinary runtime processes run current,candidate,candidate,current,
on CPU2 with the monitor on CPU0, .NET10.0.8, SDK10.0.204, AVX512/AVX2/FMA enabled,
and no DOTNET/COMPlus/LOKAD overrides. Each case has60warmup and60measured samples.
Each sample runs batch=max(1,min(1024,65536//max(1,output_elements))) complete
public CPUExecutionProvider.Where calls returning complete OpResult values. The sample includes all validation, broadcasting
or mask scan, output allocation, computation, and assignment to a preallocated
OpResult array. Setup, fixture loading, independent oracle and correctness checks
are outside timing. No forced collection, direct helper calls or reflection
occurs in the timing path. GC pauses and every measured clock remain included.

Keep every sample in a flushed clocks.jsonl journal and every setup in a flushed
setups.jsonl journal, even if a later check fails. At the last warmup and measured
sample, every OpResult must have exact success metadata and every output in the batch must match the qualified bits and coordinate
oracle, have the expected shape and its own tensor object. All input backing
stores/guards must remain unchanged. Hold a warmup output across all measured
calls; mutate a final output after timing and check the held output, other final
outputs and inputs remain unchanged. One consumer build is copied unchanged to
both product runtimes. The old numerical entry is retained but never called.

Normalize each sample as ticks/(Stopwatch.Frequency*batch), average all60measured
samples per process, then average the two processes equally per role. Aggregate
scores sum per-case means, without weighting by calls or element counts.
Require all252repeatability controls: each role's all122,target6,other_uniform42,
fallback74 sums max/min<=1.10, and each case max/min<=1.20. Require every case's
candidate/current<=1.05, the six actual targets' sum ratio<=0.90, and the maximum
candidate target sum strictly below the minimum current target sum. Other
uniform and fallback gains cannot supply the target improvement. No unchanged
failed-screen retry. Retain58,560sample clocks,29,280measured clocks and488setups.

Run locally with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/provider-where-screen/test_score.py
    tests/parakeet/provider-where-screen/run.py prepare
    tests/parakeet/provider-where-screen/run.py stage
    tests/parakeet/provider-where-screen/run.py launch
    tests/parakeet/provider-where-screen/run.py observe
    tests/parakeet/provider-where-screen/run.py collect
    tests/parakeet/provider-where-screen/audit.py

Prepare once and freeze tools before staging; refuse existing namespaces.
Observe only a launched, unclosed lane; collect only after all PID/birth owners
are terminal. Artifacts use parakeet-provider-where-screen-amd-20260924 locally and
/dev/shm/lokad-parakeet-provider-where-screen-20260924 on AMD. Preflight requires
12GiBavailable/3GiBtmpfs; preserve8GiBRSS,1GiBremainingmemory/tmpfs,900s/job,
fourhours/campaign,1GiBoutput/job,2GiBtotal artifact limits. Audit validates
every identity, file, journal, resource sample and unmodified setup. A closed
campaign may be technically valid but performance-rejected; inspect the separate
performance_admitted field. Do not turn rejection into benchmark promotion.

Admission only permits the subsequent full Parakeet/native/public/application,
shared/e5/Pyannote, actual-root, complete-suite and independent-package gates.
BENCHMARK.md continues to report the fully qualified selected release meanwhile.
