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

The completed initial capture used
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

The original capture and all later stages are now terminal. Do not rerun their
commands against the existing artifact namespaces. Completed follow-up tools:

- `probe_native.py` checks the installed library identity, symbols and perf access.
- `graphs.py` serializes original-output optimized graphs without inference.
  The initial memory-preflight refusal and successor's completed encoder export
  are retained. `--remaining` completed only the smaller decoder/frontend graphs.
- `retire_remote_profiles.py` removes three verified VM trace duplicates; complete
  local traces and collection archive remain. Serialized weight sidecars are
  separately hashed and retired as owned scratch after their sessions close.
- `sample_native.py` captures the unchanged native application with perf and
  collects it after every owner is terminal. `inspect_hot_native.py` reads the
  exact loaded ELF's disassembly. `match_native_kernel.py` assembles one pinned
  Microsoft source routine and proves its bytes match; this does not replace
  the runtime used for inference.
- `audit_native.py` reconciles raw and exported samples, request intervals,
  instruction addresses, process ownership and resource limits. It preserves
  auxiliary-thread samples and raw instruction addresses when unwinding is absent.
  `test_native_records.py` covers those cases and lost/truncated raw records.

The exact auditor used for the closed sample capture is retained as
`artifacts/parakeet-ort-native-samples-20260924/audit-final.py`, matching the
closure's auditor hash. The checked-in helper additionally closes its input
stream explicitly; this housekeeping does not change sample attribution.

The native sampler uses 199 Hz user CPU samples, a 4,096-byte DWARF stack snapshot
and the monotonic clock. Only perf and its launcher use sudo; the application
runs as the original user on CPU2, monitoring/collection on CPU0. No system
security setting changes. Prospective limits were 8 GiB available/2 GiB tmpfs
before launch, 6 GiB owned RSS, 1 GiB remaining memory/tmpfs, 512 MiB output and
300 seconds, with a separate 60-second export limit. All checks pass.

[Findings and limitations](../ort-diagnosis-results/README.md) include overhead,
source/binary identities and the still-missing matched Lokad attribution.

ORT's documented profiling API is described in
[Profiling tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html).
