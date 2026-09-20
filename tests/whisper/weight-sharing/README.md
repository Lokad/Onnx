# Private Whisper decoder weight sharing

This experiment shares exactly equal FP32 weight storage between the two decoder
graphs owned by one transcriber. Tensor wrappers, names and shapes stay separate;
there is no global cache or sharing between transcribers. General graph ownership
and production source remain unchanged.

The [local proof](local-results-20260920.md) passes 3,101 backend tests and verifies
635,187,200 fewer unique initializer payload bytes in the actual prepared decoder
pair. Complete initializer hashes and captured graph/packing metadata match before
and after. The inspection performs no inference and does not claim an equal RSS
reduction. AMD application, numerical and resource qualification remains pending.

Source templates are `WhisperDecoderWeights.cs` and `WhisperDecoderWeightsTests.cs`.
`prepare.py` copies only verified source files from the separate buffer-reuse
prototype, patches the private transcriber, builds the CLI and runs backend tests.
`inspect_models.py` builds a diagnostic consumer and bounds its local process by
time, affinity and memory. `close_local.py` audits the source, actual tested DLLs,
model census, unchanged initializer values, all resources and terminal identity.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root. These completed
tools create new evidence and must not overwrite existing artifacts. All source,
initial build errors, warnings, tests and model inspection records remain in
`artifacts/whisper-weight-sharing-20260920`. The preceding buffer-reuse VM campaign
is closed: all 100 requests and resource checks passed.

`prepare_consumer.py` adds read-only initializer hashes and backing-array identity
checks to that closed public consumer. `deploy.py` freezes the tested binaries,
source, models, runtime and prospective protocol before a new AMD launch. The
schedule remains twenty conformance requests followed conditionally by eighty
endurance requests, with unchanged resource and allocation limits. Both processes
must prove 635,187,200 logical shared bytes, actual shared storage and unchanged
complete decoder initializer snapshots. Hashing occurs outside public call timers.
Three protocol tests accept valid metadata and reject thirteen corruptions.

Use `observe.py` to inspect that existing launch and `collect.py` only after all
actual process births are terminal. `audit.py` checks successful complete runs;
failure evidence must be retained separately. These requests qualify finite
application and memory behavior, not matched Microsoft ORT latency.
