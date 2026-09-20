# Selected natural-case Whisper encoder traces

This diagnostic adds intermediate observations for three cases selected by the
closed full-corpus frame analysis, with both feature sources and the first-case
repeat. It preserves all 41 exposed outputs per call: eight finite workers,
sixteen calls and 656 arrays. Selected cases cannot replace full qualification.

The existing trace model is checked byte-structurally against the original:
only its outputs differ. The same original core, model weights, feature arrays
and actual native runtime bytes are reused. Final trace outputs are compared
with the exact unmodified arrays already reproduced in the closed full-corpus
campaign. Effects from exposing outputs are retained separately from engine
differences, because those outputs can prevent optimizer fusions.

Build `Probe.csproj` with `FrozenProductDirectory` pointing to the original
`artifacts/asr-labeled-20260919/managed-bin`, Release output `<artifact>/bin` and
`--tl:off --nologo -v minimal`; retain `build.log`. Run `check_serialization.py`
with `--artifact <directory>` (tiny graph load/serialization only, no inference),
then `test_audit.py`, retaining `unit-tests.log`. Commit tools before
`prepare.py --artifact <directory>` freezes their bytes.

`run.py --artifact <directory>` starts the fixed eight-worker schedule. Poll the
original coordinator identity/session until terminal. Then run
`audit.py --artifact <directory> --output <directory>/audit.json`, followed by
`close.py --artifact <directory>`. Inspect complete findings and independently
verify the final inventory/report/births. Successful writers are single-use.

Managed hashes and saves validated canonical dense spans without copying every
trace tensor. Native saves its optimized graph with external initializer data
using the [official serialization options](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h).
The installed ORT options are exercised before the real trace. These serialized
graphs are evidence for the observed node census, not portable deployments.
Allow at least 25 GiB free disk for all output arrays and native graph snapshots.

Workers inherit CPU 2; the coordinator uses CPU 0. Keep the same 8-GiB sampled
RSS, 10-GiB available preflight, 1-GiB available during work and 1,800-second
guards. Every observed process has a PID/creation-time identity. No forced GC,
profiler, runtime override, model download or additional VM load is used.
