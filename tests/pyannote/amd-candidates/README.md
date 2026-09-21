# AMD qualification of the pyannote candidates

This lane compares current production with the isolated portable convolution/LSTM
candidate and its AVX-512 convolution row-sharing successor. Pyannote has first
priority; complete Parakeet trajectories are a required regression gate. New
Whisper research is deferred.

The user has authorized the VM. The existing frozen e5 controller owns its two
phases until its actual processes finish; this runner checks that handoff before
writing to the VM. It never terminates or relaunches e5.

The immutable offline payload is in
`artifacts/pyannote-amd-candidates-v2-20260921`. It contains three separately
pinned Core assemblies, identical Data assemblies, original complete application
consumers, the row candidate's source and 23 cached NuGet packages. The graph
consumer only changes its platform guard to allow Linux. Source, generated
assemblies, retained models and native dependencies are checked by SHA256.

Local preparation passes nine offline restore/build/instruction-comparison
commands. Its independent audit checks 1,335 archive entries, 401 source files
and 3,795 Core/Data methods. This does not qualify AMD hardware execution.
The initial missing-SourceLink restore failure remains preserved separately.

From the repository root, the local runner tests are:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/pyannote/amd-candidates -p test_*.py -v

`prepare_execution.py` freezes a separate execution bundle without changing the
payload. `finish.py` waits for the existing e5 controller, then stages, launches,
observes, collects and audits one campaign. Commands are:

    C:/Python313/python.exe -X utf8 -B tests/pyannote/amd-candidates/prepare_execution.py
    C:/Python313/python.exe -X utf8 -B tests/pyannote/amd-candidates/finish.py

Each command creates a new artifact and refuses to overwrite it. Do not rerun a
started controller. Inspect `artifacts/pyannote-amd-execution-20260921/controller/state.json`
and actual PID/birth identities first. After deployment, read-only observation is:

    C:/Python313/python.exe -X utf8 -B tests/pyannote/amd-candidates/transport.py observe

The VM destination is `/dev/shm/lokad-pyannote-candidates-20260921`. Source
builds, package extraction, intermediate files and logs stay in temporary memory
storage. Existing model and native package directories are read-only. Offline
AMD builds use SDK 10.0.204; consumers use .NET 10.0.8 with normal runtime
settings. All numerical workers inherit logical CPU 2 before their runtime
starts; the supervisor runs on CPU 0. The campaign permits four hours overall,
one hour per worker, less than 12 GiB aggregate worker RSS, at least 1 GiB
available memory/free temporary storage, and at most 2 GiB campaign files.

The campaign first rebuilds and compares Core/Data instructions, then runs full
backend/tensor tests. Three specific AVX-512 tests must execute successfully;
a skip is not qualification. Each of the three cores must then pass all 18
pyannote output tensors and 16 full public requests, followed by all 784 Parakeet
arrays / 3,090,494 values and its public/rejection/recovery cases. The scaled
native numerical threshold stays at `1e-4`. An independent native pyannote
public conformance process must also pass before timing begins.

Timing uses eight fresh processes in production, portable, rows, ORT, ORT, rows,
portable, production order. Each runs one warmup plus three measured passes of
the four dialogue fixtures: 128 calls, 96 measured and 32 warmup. All features,
graphs, clustering and owned results remain inside the original application
timers. Model loading, file access and external validation remain outside.
Microsoft ORT 1.29.0 uses one intra/inter-op thread and sequential execution.
Every sample and process mean is retained; these are descriptive comparisons.

Failures stop advancement, retain diagnostic arrays and logs, and are collected
without automatic inference retries. Only owned worker identities may be killed
on a resource failure. Collection streams evidence to the workstation, without
writing a second results archive on the VM. The local auditor recomputes tensor
and public checks and timing means from integer clocks. It does not promote a
candidate or edit `BENCHMARK.md`: publish the audited results and assess remaining
meeting/model/package qualification before integrating production source.
