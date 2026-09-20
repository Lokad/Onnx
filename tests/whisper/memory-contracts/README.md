# Whisper memory ownership across public request transitions

This private candidate consumer extends the original recording qualification with
two overlapping non-silent short requests on the same transcriber, post-work
invalid-input and cancellation checks, and a successful short recovery request.
All actual outputs and PCM are retained and checked. Complete decoder initializer
hashes and shared storage must remain unchanged across the entire sequence.

`prepare.py` verifies the closed product proof and original references, then builds
against the already tested private Core/Data binaries. `run_local.py` freezes the
consumer and local runtime, and runs thirteen completed requests plus sixteen
refusal checks under finite process, affinity and memory supervision. Preparation
and execution create new files and must not be rerun over an existing attempt.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root. Artifact:
`artifacts/whisper-memory-contracts-20260920`. These are functional and memory
checks, not latency measurements. The existing AMD weight-sharing campaign is
independent and must close before any additional workload starts on that VM.
