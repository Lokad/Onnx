# Paired managed e5 A/A measurement experiment

This experiment tests whether two identical engines can be compared by
alternating requests inside one process. The production core and its defaults
are unchanged. There is no candidate optimization or native ORT timing in the
experiment; passing the screen would only justify designing a later managed
candidate comparison.

The host loads the same bridge and core binaries into two separate
[.NET AssemblyLoadContexts](https://learn.microsoft.com/en-us/dotnet/core/dependency-loading/understanding-assemblyloadcontext).
Private assembly/type/static-field identities are checked, including a restored
diagnostic latch. Framework services and GC remain shared. Each engine owns its
graph, canonical inputs and held outputs and calls the real public Execute API
with `ExecutionOptions.Memory`.

## Fixed protocol

Four visits cover all five canonical cases: 8 tokens, 30 tokens, 30 padded to
128, 128 and 512. Each of the twenty fresh processes creates two identical
engines. Case order and creation order reverse on odd visits. Each arm receives
30 cumulative Execute seconds of alternating conditioning. All conditioning
calls are retained.

Each process measures 64 adjacent pairs, balanced 32 AB and 32 BA, plus 32 solo
calls per arm while both graphs remain resident. A portable 32-bit shuffle uses
seed `20260920 + 100*visit + caseIndex`, recurrence
`state = state*1664525 + 1013904223 (mod 2^32)`, and descending Fisher–Yates
indices `state % (i+1)`. Solo blocks precede pairs on odd visits and follow them
on even visits. This gives 3,840 measured calls in total.

Execute and enclosing Reset-plus-Execute ticks are recorded on every call,
alongside allocation and GC-count deltas. There is no forced GC, profiling,
runtime override, sample deletion, fitted drift correction or adaptive stopping.
File IO, validation and output copying remain outside timing. Measurement storage
is allocated before the timed phase, and serialization occurs afterwards.

For each case and both boundaries, the prospective screen requires aggregate
B/A means within 0.5% of one, each visit within 1%, the AB/BA ratio contrast
within 1%, and each arm's paired/solo ratio within 5%. Every sample counts,
including GC tails. These are eligibility limits for small managed contrasts,
not confidence intervals or a claim of sample independence. The solo blocks do
not replace a separately isolated deployment comparison. Native/managed process
interactions and ORT parity remain outside this lane.

## Functional and resource checks

All complete before/after outputs are retained and independently compared with
the existing pinned native ORT 1.23.2 outputs at scaled error `1e-4`. A/A and
before/after outputs must match bits. Inputs and the first returned outputs must
survive later resets/calls unchanged. Product binaries are pinned to source
`8732831`, core `05884cfd`; no native ORT library may load in the managed worker.

AMD workers inherit CPU 2 before CLR startup; supervision uses CPU 0. Every
worker has 8 GiB RSS and 600-second guards with at least 1 GiB system available
memory. Process births, affinity, memory, CPU use, foreign-process snapshots and
guest `/proc/stat` are retained. Execution failures stop the campaign; timing
screen failures are assessed only after the complete fixed schedule.

The local eight-token smoke uses one second of conditioning and four paired/
four solo calls per arm. It checks actual isolation, native outputs and held
ownership, and supplies no AMD timing conclusion. The first smoke refused an
incorrect dimension-width encoding in the probe's input hasher before model
execution. The corrected smoke is retained separately under
`artifacts/e5-paired-aa-v2-20260920`; the failure remains under the original name.

## Tools and execution

From the repository root, build `Bridge.csproj` with
`-p:FrozenProductDirectory=<qualified recording-bin directory>`, then
`Host.csproj`, both Release into the same fresh `bin` directory. Use
`--tl:off --nologo -v minimal`. The bridge references only the exact frozen core
and Protobuf binaries. `prepare_inputs.py` verifies a closed source archive and
native fixtures and writes the five inputs without inference.

Use the existing Python environment at
`artifacts/asr-labeled-20260919/venv/Scripts/python.exe`:

    python -B tests/e5/paired-aa/local_smoke.py --artifact <new-artifact>
    python -B tests/e5/paired-aa/audit.py --artifact <artifact> --smoke --output <artifact>/smoke-audit-final.json
    python -B -m unittest discover -s tests/e5/paired-aa -p test_audit.py -v
    python -B tests/e5/paired-aa/prepare.py --artifact <artifact>
    python -B tests/e5/paired-aa/vm.py deploy --artifact <artifact>
    python -B tests/e5/paired-aa/vm.py poll --artifact <artifact>
    python -B tests/e5/paired-aa/vm.py collect --artifact <artifact>
    python -B <artifact>/payload/audit.py --artifact <artifact> --output <artifact>/audit.json

The current VM path is fixed to `artifacts/e5-paired-aa-v2-20260920`; change it
prospectively for a separately named campaign. Successful writers refuse existing
outputs and run once; only polling is repeatable. A timeout while observing a
job does not terminate it. Verify the same PID and birth before any recovery.
Preserve every failure and do not reinterpret a functional smoke as A/A timing
acceptance.
