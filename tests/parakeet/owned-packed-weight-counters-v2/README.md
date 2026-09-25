# Count actual preparation traffic before timing M76

This is the prospective accounting protocol following complete native/public
model qualification atbb1a758c. Its preparation verifies that closure and the
actual87-weight census577a07a6 before any consumer build or model execution.

Use the already compiled selected M73 and corrected M76 Core/Data DLLs. Construct
ParakeetTranscriber normally, access its frontend and encoder, and execute the
same public-corpus input dictionaries and memory options. The consumer deliberately
stops after encoding: complete public decoder checks belong to the preceding
full-model qualification. No Core/Data observer build or modified kernel is needed.

Across all20 clips, require exact selected/candidate frontend and encoder hashes,
immutable audio/features, independently held encoder outputs, actual matrix
dimensions and all2856 ordered profile records. Extract only96 feed-forward
stage counts per request and the existing graph copy/scratch counters. Discard
profiler durations: no measured latency is scored by this diagnostic.

The frozen prediction must be87 fewer16MiB scratch packs per clip. Seven lengths
(167,89,157,83,61,151,169) predict one16MiB reconstruction for each owned weight;
the other13 lengths predict none. Thus1740 packs disappear and609 reconstructions
remain per corpus:29,192,355,840 fewer cumulative scratch bytes and10,217,324,544
additional cumulative copy bytes. Check the exact per-clip counter differences
and CopyY stage locations, including all nine untouched prepared weights.
These totals describe cumulative traffic, not retained memory or performance.

Run selected/candidate in separate processes in normal and AVX512-disabled modes.
Retain the census guards (11GiB free RAM,2GiB tmpfs,12GiB RSS,900seconds per job,
1GiB remaining RAM/tmpfs,CPU2 workload/CPU0 monitor) and128MiB total output cap.
Build only this consumer against binary references on the VM. Verify identities,
the unchanged product binaries and exact assembly dependencies before and after.

Use Python3.13 -X utf8 -B with run.py prepare, stage, launch build, observe build,
collect build, then review.py build. After that review passes, use run.py launch
capture, observe capture, collect capture, then review.py capture. Tools freeze
at preparation. Preserve every failure and never repeat a completed worker or
collector. Store audit stdout outside the campaign. The collector uses the
retained transport with tar dereferencing for the two hardlinked runtime trees.
Artifacts use parakeet-owned-packed-weight-counters-v2-amd-20260925; the VM uses
/dev/shm/lokad-parakeet-owned-packed-weight-counters-v2-20260925.

The initial build passed compilation but failed its zero-warning review because
the Linux guard was hidden inside a Require call (CA1416). V2 uses an explicit
platform rejection. It preserves the entire original build and changes no
accounting, numerical, resource or product requirement. No original capture ran.
