# Actual Parakeet prepared recurrence qualification

This prospective lane verifies the unchanged original decoder and every complete
LSTM call captured in closure28c7afe4. It compares selected Core672e5f30 with
candidate Core3c23b44a using one newly built consumer, in normal and AVX512-disabled
processes. Candidate build refusal b2d46d3b is preserved; separate review1b7f8e21
accepts its exact compiler identifier changes. Contracts aacf2dbe already passed
150 cases per mode. No product is rebuilt here, and no performance claim is made.

The original decoder retains three matrix preparations (25,246,720 bytes).
Candidate adds four exact recurrent transposes (26,214,400 bytes), yielding
51,461,120 bytes under the unchanged 64 MiB cap. Each actual one-node graph retains
two recurrent arrays (13,107,200 bytes); selected retains zero. Every owned array,
source binding, transpose element, repeat preparation, invalidation/rebuild and
fresh context map is checked. Original graph nodes, outputs and all 13 initializer
references and bytes remain unchanged. Cached matrices retain the closed residency
digests; recurrent digests are independently computed from the canonical model.

Test-only reflection fills recurrent prepared arrays with NaN, requires original
decoder state outputs and each complete-call output to become NaN, then invalidates
and prepares again and requires exact recovery. Original output bindings stay in
place: exposing W/R as graph outputs would disable candidate admission. No product
instrumentation or alternate dispatch is introduced.

Each worker repeats all 190 recorded decoder steps twice, independently carrying
accepted states, and repeats all 380 actual complete LSTM calls twice. Every
original output and decision must match the selected saved trajectory exactly;
all complete-call output bits must match capture, and their scaled error against
the closed ORT1.29.0 arrays must be <=1e-4. Both modes and both products must have
the same normalized output digest. Four workers produce 6,080 original decoder
array comparisons and 9,120 complete-call array comparisons (5,836,800 component
values). Inputs remain immutable and held outputs survive later execution,
reset and preparation invalidation. Raw arrays are deduplicated by SHA256.

From the repository root, use Python3.13 with `-X utf8 -B` to run `run.py prepare`,
`run.py stage`, `run.py launch`, `run.py observe`, `run.py collect`, then `audit.py`
in this directory. Preparation verifies all pinned predecessor closures and all
422 unchanged selected root inputs. Staging reuses immutable capture/native files
and six saved encoder arrays through hardlinks in a new `/dev/shm` namespace.
Never overwrite hardlinks. There is no encoder inference, native rerun, model
copy, dependency download or Windows build. Model parsing on Windows only derives
initializer and transpose hashes. The AMD helper uses SDK10.0.204, runtime10.0.8,
CPU2 before CLR startup, monitoring on CPU0, and the existing offline feed.

Preflight requires 8 GiB available memory and 2 GiB free tmpfs; each job is bounded
by 4 GiB owned RSS, at least 1 GiB available memory/tmpfs, 128 MiB output and
900 seconds. Whole campaign artifacts are bounded by 768 MiB and duration by four
hours. Every process/thread must remain on CPU2. Parent PID/birth owners must be
terminal; collection verifies every owned PID/birth is dead. Raw monitoring,
consumer/product identities, all arrays and independent NumPy recomputation are
retained. Never observe a closed campaign or silently rerun a failed lane.

Passing this lane permits full native/public trajectory qualification, followed
by prospectively frozen complete-call timing and the unchanged full application
selection. It does not promote the candidate or change BENCHMARK.md.

The first consumer stopped on the selected standalone graph because inputs were
null without type/shape descriptors. Refusal closure `bc2ce801` preserves all
outputs and owners; no candidate ran. This revision supplies float tensors with
the exact captured input/output shapes. Products, fixtures, gates, execution
order and resource bounds remain unchanged. The helper alone is rebuilt.
