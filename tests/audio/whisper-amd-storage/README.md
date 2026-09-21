# Storage-corrected AMD Whisper comparison

This separately declared comparison addresses the [previous root-disk failure](../whisper-amd/disk-failure-20260921.md).
It does not change inference or numerical acceptance. New assets, binaries,
evidence and temporary files use `/dev/shm`; actual mount/device checks prevent
silently falling back to the root filesystem. Collection streams directly to
local disk without constructing a second VM archive. Model weights and numeric
libraries remain read-only at their verified existing paths.

Both original twenty-call conformance gates are fully rechecked and reused.
Four fresh timing processes run managed, ORT, ORT, managed. Each runs twenty
warmups and sixty measured calls. None of the failed campaign's timing data is
reused. The original CPU2, single-thread, runtime, memory and time limits remain;
the output filesystem requires 3 GiB free before launch and 512 MiB during work.
Root free space is retained separately in every resource sample.

Run from the repository root with `C:/Python313/python.exe -X utf8 -B`:

1. `tests/audio/whisper-amd-storage/test_contract.py`
2. `tests/audio/whisper-amd-storage/prepare.py`
3. `tests/audio/whisper-amd-storage/finish.py`

The completion controller waits for the actual existing e5 controller to finish.
Staging independently refuses a live or failed predecessor and verifies all
recorded remote process births are terminal. It then stages and launches once,
observes without restarting, collects complete or failed evidence, and runs
success-only audit/report/verification when every timing worker completes.
Do not invoke individual writers while this controller is active.

Preparation makes no VM changes and does not rebuild qualified binaries.
Artifacts live under `artifacts/audio-whisper-storage-20260921`; controller state
and logs are separate under `artifacts/audio-whisper-storage-finish-20260921`.
Reports and BENCHMARK.md update only after complete validation, including an
independent reconstruction from integer timer ticks. Preparation is not a result.
An observation timeout does not establish that inference stopped; inspect the
original process identities. Never overwrite existing manifests or restart a
worker to replace inconvenient data.

Temporary apt/PackageKit masks from the earlier incident must be restored after
the entire benchmark window. This runner does not restore them during inference.
