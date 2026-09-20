# Whisper collection diagnostic

This is a prepared diagnostic consumer, not a benchmark configuration or a
production change. The normal-runtime AMD campaign stopped during request 17
at its unchanged available-memory guard; its [closed failure](../../audio/amd-comparison/resource-failure-20260920.md)
remains the observed result.

The consumer retains the same models, PCM, options and exact public-result checks.
It additionally records heap/collector/process memory around each request and
explicitly requests blocking generation-2 collection after requests 8, 16 and 20.
All input arrays, earlier actual output objects and the transcriber remain held
and are checked after each intervention. `GC.GetGCMemoryInfo` fields describe the
last completed collection; they are not live process-memory measurements.

The consumer requests `compacting: true` through `GC.Collect`; it does not set
large-object-heap compaction policy or change environment/runtime settings.
Managed bytes reclaimed, committed heap and process RSS are separate observations.
A reduction would show reclaimability under this intervention, not identify every
retained object, prove the absence of a leak or qualify normal repeated operation.

Run preparation from the repository root:

    C:/Python313/python.exe -X utf8 -B tests/whisper/memory-collection/prepare.py

Preparation is already complete in
`artifacts/whisper-memory-collection-20260920`: SDK 10.0.204, zero build warnings
or errors, 44 pinned files and unchanged private product binaries. The command
refuses an existing output directory. No new VM execution has occurred.
Deployment, the bounded process supervisor and independent result/resource audit
must be completed under the focused plan before drawing a memory conclusion.
