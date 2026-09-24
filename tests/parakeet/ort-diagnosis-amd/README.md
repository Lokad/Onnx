# Diagnose the exact Parakeet ORT baseline

This diagnostic keeps the installed ORT 1.29.0 binary, original model outputs,
twenty-clip PCM corpus and native decoding consumer. It adds one reviewed hook
after that consumer loads its pinned adapter. All existing numerical, repeated
result, input ownership and held-result checks still execute unchanged.

`observer.py` wraps the existing application and graph-call methods, recording
frontend, encoder and decoder wall/CPU clocks, call counts and tensor shapes.
It does not retain intermediate tensor arrays. The control uses the original
session settings; the profile adds only ORT's built-in profiling and output prefix.
Both use one warmup and three measured passes, 80 complete requests and 4,960
graph calls. The extra timing and shape observation affects both diagnostic modes.

The original consumer's reported hash identifies the original source. The
diagnostic wrapper and its exact AST insertion are separately recorded; the
instrumented consumer is not represented as an unchanged executable. Stripping
that single insertion must reproduce the entire original syntax tree exactly.

Preserve every raw clock and event. Reconcile session runs to graph calls before
attributing operators to requests; warmups remain separate. Profile clocks do
not replace the release benchmark. Compare the control with the original native
application, then quantify profile overhead. Graph serialization and native
sampling are separate diagnostic stages, chosen after the initial attribution.

This implements the user's request to select subsequent improvements from
observed ORT behavior rather than enumerating speculative kernel variants.
The detailed plan is `.agent/m65-parakeet-ort-diagnosis-20260924.md`.

After the M64 application campaign is terminal, collected and audited, use
`C:/Python313/python.exe -X utf8 -B` from the repository root with this directory's
`test_attribution.py`, then `run.py stage`, `launch`, `observe`, `collect`, and
`analyze.py`, each separately. The local evidence directory is
`artifacts/parakeet-ort-diagnosis-amd-20260924`; the VM namespace is
`/dev/shm/lokad-parakeet-ort-diagnosis-20260924`. Refuse existing destinations;
never append observations after closure or retry inference to fix an analyzer.

The fixed sequence is control then profile. Both execute all 80 requests. All
target threads use CPU2, monitoring CPU0. Require 12GiB available and 3GiB tmpfs
before each process; cap owned RSS at 12GiB, total retained output at 1GiB and
each worker at 900seconds. Keep at least 1GiB available memory/tmpfs. Preserve
process births, resource samples and foreign CPU accounting with its documented
limitation for short-lived processes. Collection requires all owners terminal.
No native library, product DLL or model is rebuilt or copied.

ORT's documented profiling API is described in
[Profiling tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
