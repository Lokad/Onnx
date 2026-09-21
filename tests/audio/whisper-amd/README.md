# Integrated Whisper versus ORT on AMD

The existing complete-application consumer measures the integrated memory changes
against ORT 1.29.0 on AMD EPYC 9V74, CPU 2, .NET 10.0.8. Product source is
`0f86c5d`; the actual assemblies come from its source-equivalent qualified archive.
No arithmetic or request policy changes enter this comparison.

One fresh managed process must pass all twenty conformance requests. The native
conformance gate reuses twenty earlier successful requests only after checking
unchanged model, PCM, native code/libraries, every output and resource sample.
The earlier managed resource failure remains recorded separately.

Four fresh timing processes run managed, ORT, ORT, managed. Each runs one full
warmup and three measured passes over twenty clips (213.265 seconds of audio).
Timing covers features, neural inference, decoding and owned results. Loading,
file access and validation are excluded. Normal collection and unchanged limits
apply: 3,600 seconds, 14 GiB RSS, 1 GiB available memory, 32 MiB disk per worker;
13 GiB available and 64 MiB disk before launch. The complete campaign has four hours.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` with scripts in
this folder: `prepare.py`, `stage.py`, `observe.py`, then terminal `collect.py`,
`audit.py`, `close.py`, `verify.py`. Preparation and staging each run once per
new artifact; observation is read-only. Never restart an observed worker or
overwrite completed evidence. On failure collect the intact records before
diagnosis; success-only reporting does not accept failed campaigns.

Results are descriptive application measurements, without a calibrated parity
claim. Existing encoder/logit numerical failures remain open. Raw source, binaries,
inputs, timings and resource evidence live in `artifacts/audio-whisper-amd-20260921`.
