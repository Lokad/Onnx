# Refresh native attribution after pointwise release qualification

Actual root/package admission is bound at `fc116763`; source `c28b3848` is
qualified and committed. Built Core is `a6f7d9f9`, Data `1ba343fd`. The measured
application is closure `505da6ab`, Core `7cac6788` / Data `dd56902f`, with
45.332653922 seconds versus matched ORT 39.232132721. The actual normal build
matches all 3,288 Core and 697 Data methods, flags and public metadata.

Reuse ort-diagnosis-amd's observer, worker, transport and complete auditor.
Only application/output locations and qualification bindings change. Keep the
original control/profile pair, eighty requests per process (twenty warmups and
sixty measurements over twenty clips), original graph outputs, decoding,
numerical/ownership checks, ORT 1.29.0, one thread and graph optimizations.
Reconcile all 4,960 graph calls per process, every node event and remaining time.
Report control/unprofiled and profile/control ratios separately. Profiling is
attribution; its clocks cannot replace the BENCHMARK.md comparison.

Keep CPU2 compute / CPU0 monitoring, 12 GiB available RAM / 3 GiB free tmpfs
before each process, 12 GiB maximum RSS, 1 GiB remaining RAM/tmpfs, 1 GiB output
and 900 seconds/process. No other VM workload may overlap. Preserve every failure.

After binding actual successful root qualification and freezing the adapter,
prefix with C:/Python313/python.exe -X utf8 -B from repository root:

    tests/parakeet/pointwise-tail-ort-profile-amd/run.py review
    tests/parakeet/pointwise-tail-ort-profile-amd/run.py stage
    tests/parakeet/pointwise-tail-ort-profile-amd/run.py launch
    tests/parakeet/pointwise-tail-ort-profile-amd/run.py observe

Follow the same owner to terminal, then collect and analyze.py once, keeping
audit output outside the artifact. Pair this with the fresh managed profile.
Use exact-export membership and pinned ORT source to diagnose the largest
remaining measured excess before selecting one new change.

Local: artifacts/parakeet-pointwise-tail-ort-profile-amd-20260927.
VM: /dev/shm/lokad-pwt-ort-profile-20260927.
