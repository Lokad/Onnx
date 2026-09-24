# Complete Parakeet correctness for prepared recurrence

Run after reviewed build `1b7f8e21`, focused contracts `aacf2dbe` and actual
decoder/call qualification `20b3bc4b`. Selected Core `672e5f30` / Data `065b7a7f` and candidate
Core `3c23b44a` / Data `cc37b19e` use the same qualified consumers:
TranscribeReplay `335ca09d` and AudioBenchmark `7eca033a`. No product or consumer
is rebuilt, and existing models, audio and native reference tensors are reused.

Eight fresh CPU2 processes run selected native/public then candidate native/public
in ordinary mode, followed by the same four jobs with AVX-512 disabled. Each
native worker must pass all 784 arrays / 3,090,494 values at the unchanged ORT
scaled-error bound of 1e-4, plus transcription decisions, rejected requests,
cancellation and recovery. Candidate tensors must match selected bytes exactly
within each mode. Each public worker checks all twenty clips, input hashes,
held-output ownership and exact complete selected results.

Only `DOTNET_EnableAVX512=0` is allowed for disabled-mode jobs; ordinary jobs
have no managed runtime overrides. Both consumers record those raw settings.
The wrapper requires the exact expected dictionary. Only after this check does
it pass a copy with an empty flag dictionary to the original public auditor,
preserving every original numerical, result and ownership assertion. The native
and public auditor files are byte-identical to their qualified originals.
Raw flags and supervisor environment records are retained and independently audited.

Bounds remain 12 GiB available / 3 GiB tmpfs before each worker, 12 GiB owned RSS,
1 GiB remaining memory/tmpfs, 1 GiB output per job, 2 GiB artifacts, 1,800 seconds
per worker and four hours total. All worker threads use CPU2, monitor CPU0.
No Windows build or inference. Existing immutable files are hardlinked; product
entries are unlinked before replacement, never overwritten in place.

Freeze tools, then use `C:/Python313/python.exe -X utf8 -B` from the repo root:

    tests/parakeet/prepared-recurrence-models-amd/run.py prepare
    tests/parakeet/prepared-recurrence-models-amd/run.py stage
    tests/parakeet/prepared-recurrence-models-amd/run.py launch
    tests/parakeet/prepared-recurrence-models-amd/run.py observe
    tests/parakeet/prepared-recurrence-models-amd/run.py collect
    tests/parakeet/prepared-recurrence-models-amd/audit.py

Refuse existing output `artifacts/parakeet-prepared-recurrence-models-amd-20260924`
and VM `/dev/shm/lokad-parakeet-prepared-recurrence-models-20260924`. Collect only
terminal PID/birth owners. Preserve any failure; no application timing may follow
an unqualified result. This lane establishes correctness, not a speed comparison.

The original build-wrapper refusal `b2d46d3b` is preserved and accepted only
through separate compiled-scope review `1b7f8e21`; its products are reused without
a rebuild. The actual decoder helper refusal `bc2ce801` is also retained; its
corrected qualification `20b3bc4b` passes all decoder and complete-call arrays.
