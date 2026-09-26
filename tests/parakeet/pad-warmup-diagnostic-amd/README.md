# One padding diagnostic after an elapsed-time warmup

Use unchanged selected Core 521bae17 and rejected M47 Core 060950a1, ordinary
.NET 10.0.8, CPU 2 for requests and CPU 0 for collection/export. No product build,
new optimization or performance admission. The prior screens remain rejected.

The retained consumer gains one priming method and its Main calls it before the
otherwise identical twelve-case 600/180 workload. Priming repeats the entire
same 780-call census, with the same oracle, immutable-input and independent-output
checks. It stops at the first complete census ending at least ten seconds after
the first census ended. Bounds are sixteen rounds and 180 seconds; exceeding
either fails. Every clock is retained. Separate marker IDs 3/4 identify prefix
calls; original IDs 1/2 remain unchanged. The ten-second interval was selected
before running, from the retained compilation histories, never from a speed gate.

The specific question is whether the three fallback differences remain after
ordinary whole-method optimization. Require fully optimized Pad and PadCore
loads before the original measured suffix in both roles, and PadDispatch in the
candidate. Report all later relevant loads. Availability is not per-instruction
execution proof. A missing state condition makes this probe inconclusive; do
not extend it or rerun it to get a passing ratio. Even favorable clocks cannot
substitute for the original product correctness, repeatability, shared-model and
complete-application qualification.

The retained transport, supervisor, exporter and suffix auditor are reused.
Build only the generated consumer on the VM, using SDK 10.0.204 and the offline
feed. Tools and inputs freeze at preparation. Verify the prefix independently
before invoking the original suffix reconciliation; preserve every raw event.

Prospective resource bounds: 4 GiB available RAM and 1 GiB free tmpfs before each
job; 2 GiB owned RSS; at least 1 GiB remaining RAM/tmpfs; 256 MiB output per job,
512 MiB total; 900 seconds per job and four hours overall. The old short trace
peaked at 333 MB; the new 2 GiB ceiling covers the bounded prefix and exporter.
These caps are fixed for this separate diagnostic. Current VM read-only preflight
found 11.88 GB available and no runtime process. Do not alter limits after launch.

From this directory, run `C:/Python313/python.exe -X utf8 -B run.py` with
`prepare`, `stage`, `launch`, `observe`, `collect` as separate actions, then
`audit.py`. Local namespace: artifacts/parakeet-pad-warmup-diagnostic-amd-20260926.
Remote namespace: /dev/shm/lokad-parakeet-pad-warmup-diagnostic-20260926.
Do not repeat a completed capture or reuse a namespace after failure.
