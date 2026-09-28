# Refresh native attribution after transpose release qualification

Actual root/package admission 175693d3 is bound to committed source 04584fc2.
All 18 jobs, both full hardware modes and independent NuGet consumption pass.
The measured candidate
is Core c471f5d1 / Data b04aea50; Parakeet application 2e4741a6 admits a 2.420895%
matched gain, 43.459338592 seconds versus ORT 39.202810428, ratio 1.108577118.
The <=1.05 target remains open. Do not infer the remaining gap from old profiles.

Reuse the original ort-diagnosis-amd observer, worker, transport and full auditor.
Change only application/output namespaces and qualification bindings. Keep control
and profile processes, each with 80 requests: 20 warmups and 60 measurements over
the same 20 clips. Preserve all graph outputs, decoding, numerical and ownership
checks, ORT 1.29.0, one thread and graph optimizations. Reconcile all 4,960 graph
calls per process, every node event and remaining time. Report control/unprofiled
and profile/control ratios separately. Attribution clocks never replace release
benchmark scores; do not subtract observer overhead.

Keep CPU2 compute / CPU0 monitor, 12GiB available RAM / 3GiB tmpfs before each
process, 12GiB maximum RSS, 1GiB remaining RAM/tmpfs, 1GiB output and 900 seconds
per process. One VM workload at a time. Preserve every failure and all inputs.
No inference, staging or new model downloads belong to adapter preparation.

After actual source/package admission and source commit, bind its closed proof
to ROOT_DIGEST and freeze both profile adapters. From repository root prefix
C:/Python313/python.exe -X utf8 -B, then use this directory's run.py review,
stage, launch and observe. Follow that owner to terminal, collect once and run
analyze.py once with console output outside the artifact. Never replay.

Pair the new native capture with the new managed profile. Inspect pinned ORT
source for the largest remaining measured excess before selecting one causal
intervention. No further optimization is selected by this adapter.

Local: artifacts/parakeet-transpose-axis-ort-profile-amd-20260928.
VM: /dev/shm/lokad-transpose-axis-ort-profile-20260928.
