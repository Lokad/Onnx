# Whisper: maximum-duration speech qualification

This lane exercises `WhisperTranscriber.TranscribeRecording` on exactly 600
seconds of mono 16 kHz speech. It reuses the closed recording implementation's
core and Data DLLs and checks a new workload; it makes no product changes.

The PCM cycles a 69.455-second connected English fixture and truncates it at
9,600,000 samples. It is a constructed resource test, not independent natural
long-conversation accuracy. English is explicit, each window allows 444 new
tokens, and the request allows 256 windows.

The native oracle runs pinned ORT encoder/decoder graphs with the retained
OpenAI timestamp rules and seek code. It saves every full logits array. A
separate audit reconstructs each greedy choice and confidence from those
arrays. Managed replay checks exact application decisions, repeated-result
stability, input/output ownership, ten refusal/recovery cases, maximum silence,
concurrent silence and the existing short API. Both maximum-speech requests
must complete the full recording. Floating-point confidence differences remain
diagnostics; this lane does not establish full managed intermediate tensor
agreement.

The recorded execution uses source `6e4b16f` for preparation and `4e4cd47` for
the corrected supervision/AMD tools. Product source is `8732831`. The first
local memory preflight refused before inference; a later attempt passed the
same margin. The native worker completed, but an immediate post-exit identity
assertion failed. `recover_native.py` preserves that failure and independently
verifies all process identities absent, unchanged payload/arrays, complete
coverage and the native audit. It never rewrites old evidence or reruns
inference. The corrected supervisor checks current PID existence as well as
birth time and records a bounded ten-second exit drain.

From the repository root, use the existing Python environment under
`artifacts/asr-labeled-20260919/venv/Scripts/python.exe`. Every writer refuses
existing outputs. The completed preparation and native generation must not be
repeated into their retained directories.

The preparer creates a new one-case runner from the exact archived recording
source and builds only that executable against the frozen DLLs:

    python -B tests/whisper/maximum-speech/prepare.py --artifact <new-local-directory>

Use the frozen native source under that directory with its `--models`,
`--inputs` and `--output` arguments, supervised with `supervise.py --phase native`.
Run its retained `audit_native.py` before launching managed inference. In the
recorded attempt, the corrected supervisor is copied to
`artifacts/whisper-maximum-speech-20260919/continuation-source` and bound by
`continuation.json`. Local inference inherits CPU2 before startup, with
supervisor CPU0, 16 GiB RSS, 7,200 seconds and 1 GiB available-memory guards.
Starting headroom is at least 15 GiB.

After native audit, `prepare_amd.py --windows <local-directory> --artifact
<new-AMD-directory>` packages the identical runner and native decisions.
`vm.py deploy --artifact <AMD-directory>` verifies every transferred file and
hard-links matching prior binaries and PCM on the VM. `vm.py poll` is read-only;
`vm.py collect` runs once after every owned process is absent. AMD inference
uses CPU2, supervisor CPU0, 13.5 GiB RSS, 3,600 seconds and 256 MiB available
memory guards. Canonical model assets and unique evidence are preserved.

`audit.py --artifact <local-directory> --output <new-local-audit.json>` checks
the complete local trace. `audit_amd.py --artifact <AMD-directory> --windows
<local-directory> --output <new-AMD-audit.json>` additionally checks collection
integrity, resources and exact cross-host application decisions. The malformed
evidence tests use the retained native result; run them with:

    python -B -m unittest discover -s tests/whisper/maximum-speech -p "test_*.py" -v

Any retained API duration covers the public request, including frontend,
inference, decoding and owned output construction. Loading and external checks
are outside that stopwatch. Native generation includes validation and array
export, so its elapsed duration is not an ORT latency baseline. The dedicated
matched measurements for all three audio families remain in
[BENCHMARK.md](../../../BENCHMARK.md#audio-matched-microsoft-onnx-runtime-baselines).
