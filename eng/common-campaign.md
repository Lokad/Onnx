# One runner for released and candidate cores

`tests/Lokad.Onnx.Campaign` compiles the canonical Bench model workloads once
against the released public API. It links the same input construction,
validation, timing loops and tokenizer source for both arms. Each measured
process gets its own copy of that runner and its dependencies; only
`Lokad.Onnx.dll` changes. There is no reflection adapter in timed Execute,
or new friend assembly. One logical CPU is enforced. This page describes the
paired-process schema-1/2 producer. The prospective
[isolated e5 producer](isolated-evidence.md) runs engines separately to avoid
managed GC interference during ORT calls; it requires a new schema-3 calibration.

The new producer is `eng/run_common_campaign.py`. The older
`eng/run-l0l1-canonical.ps1` and its accounting/proof scripts remain historical
and are not used by this path. Do not use their log-only scorer invocation
for the new evidence contract. The two paths must not run concurrently.

## Prepare and check locally

From the repository root, choose a new output directory:

```powershell
pwsh -NoProfile -File eng/build-common-campaign.ps1 -Output artifacts/common-prepared
python eng/run_common_campaign.py --prepared artifacts/common-prepared --output artifacts/common-smoke --cpu 4 --kind comparison --smoke
dotnet artifacts/common-prepared/runner/Lokad.Onnx.Campaign.dll selftest
python eng/test_common_campaign.py
python eng/test_campaign_scorer.py
```

Choose a CPU that exists on the local machine. The build script archives the
released commit `4495fc68b9505b0b6fab73146bd905424588218e` and candidate HEAD
from existing local git objects, builds both cores with the current pinned
SDK, and builds one runner against L0. Override `-L0Ref`/`-L1Ref` explicitly
when comparing another accepted candidate. No fetch or worktree deletion
occurs. The archive, its digest, build logs and per-core `build.json` remain
in the prepared directory. Uncommitted core changes are deliberately absent
from a git-archive build. The common runner compiles the current local
workload sources; its actual compiled bundle is hashed.

Prepared cores and runner dependencies must stay unchanged throughout A/A
and comparison. Both output and staging directories must be new. The
supervisor verifies core/archive digests before copying and the child verifies
its actual loaded core. Prepare/build occurs before measurement, never
underneath a running campaign. The common project intentionally requires an
explicit `CoreAssemblyPath` and is not part of the ordinary solution build.

Smoke runs use one pair, two timed samples and one fixed warmup. They write
`kind=smoke`, which the scorer rejects. `--models e5 --case e5-8tok` selects
a smaller smoke workload. Scored runs use `--scope full` (default, all fifteen
cases) or `--scope e5` (five ordered e5 cases), without `--models` or `--case`.
The new producer writes schema 2 with the selected scope bound in each child
record. It requires a freshly captured schema-2 A/A with the same scope;
old schema-1 full-suite campaigns remain readable by the offline scorer.
All model and tokenizer files are local. Smoke still checks numerical
agreement, cross-core input/model identities and identical runner/runtime/ORT
binaries. It records foreign CPU but does not call a noisy machine quiet.

The supervisor pins children **before the .NET runtime starts**: Linux uses
[`taskset --cpu-list`](https://man7.org/linux/man-pages/man1/taskset.1.html);
Windows briefly narrows its own affinity during child creation, then restores
it immediately. Windows [child processes inherit that affinity](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setprocessaffinitymask).
The managed runner verifies the selected OS CPU ID again. Runtime startup
therefore sees one allowed CPU, including for worker initialization. No other
process's affinity is changed. The supported selector range is CPU IDs 0–63;
the OS must also allow the selected CPU.

## A/A and comparison on the AMD baseline

The user assigned the AMD VM exclusively to this work on September 18, 2026,
including permission to stop interfering workloads. No further coordinated
window is required. Run one measurement lane at a time, verify process identity
before stopping interference, and preserve the binaries/assets of any active
lane. A quiet guest still does not establish an uncontended hypervisor host.

```powershell
python eng/run_common_campaign.py --prepared artifacts/common-prepared --output artifacts/common-aa --cpu 2 --kind aa
python eng/run_common_campaign.py --prepared artifacts/common-prepared --output artifacts/common-comparison --cpu 2 --kind comparison --aa artifacts/common-aa/evidence.json
```

For the e5 optimization loop, add `--scope e5` to **both** commands and use
new output directories. This preserves the four-pair, 33-sample contract and
all noise/numerical/identity checks. It does not certify other model families.

A/A stages the identical L0 core on both sides. Comparison stages L0/L1.
Both use four pairs in `L0,L1 / L1,L0 / L1,L0 / L0,L1` order, with 300 seconds
between pairs and 33 samples per case. `--jit default-tiered` is the default;
`--jit full-opts` explicitly disables tiered compilation in children only.
Collect a separate preceding A/A for each regime. Full-opts results do not
describe default-tiered deployment. All other inherited DOTNET/COMPlus/LOKAD
overrides and the actual GC mode are captured, and must match calibration.

Before timing, fresh probes load both cores and native ORT, then compare
runner/runtime/native identities. Comparison also checks them, the baseline
core and current model/external-data files against the supplied A/A. A changed
SDK/runtime/runner/ORT/model aborts before the long campaign. Input identities
and numerical agreement are additionally checked from the executed workloads.
Probes have no inference timings and never become scored repetitions.

The runner requires at least 1000 ms of measured execution **per engine per
case**, a minimum of three warmup pairs and a stable last-nine-sample window
for each engine. The window's range must be <=10% of its median. It stops
after at most 1000 pairs or 60 seconds of warmup; failing to converge prevents
timing that case and records an explicit quarantine. These are preliminary warmup controls;
the scorer still checks chronological drift and fresh-process variation.
An AMD tiering/GC diagnosis and unchanged calibration are still required.

Quarantine note (September 17, first A/A): resnet50-224 warmup never converged in 60 s
(strict period-2 ~165/~205 ms alternation on both engines across 206 pairs; GC cadence under
workstation GC is the prime suspect, unproven). The runner quarantines such cases into
`cases_failed` with no rows instead of aborting the leg; the scorer reports them INCONCLUSIVE.
Per-case `gc gen0/gen1/gen2` deltas in the child log aid future warmup diagnoses.

Load, prepare, first Execute/Run, warmed public Execute, reused context and
complete request/reset diagnostics retain separate boundaries. File hashing
and process metadata capture occur outside timed regions. Cold figures are
process/startup diagnostics, not a claim of cold OS page caches. Request or
context timings never silently replace the public Execute score.

`Bench profile ... [--wall]` writes separate diagnostic schema-2 JSON with
`timingContract: "execute-only-v2"`. `repLokadExecuteMs` brackets only public
Execute; `repLokadPhases` records Reset before/after, outer profiler scope
setup/disposal, and per-repetition report aggregation separately. Profiler
collection performed **inside** Execute remains included in Execute, as does
other execution bookkeeping. Final cross-repetition report merging/serialization
is outside these timings. Adjacent unprofiled Lokad/ORT controls indicate
observer cost but lack campaign warmup/calibration and cannot score. Old
unversioned `repLokadWallMs` included Reset and report work and must not be
compared as the same metric.

## Evidence and accounting

Each completed child writes `*.process.json`. The supervisor binds it and the
raw stdout log to SHA-256, records observed PID/exit/timestamps, and assembles
`evidence.json` according to [campaign-evidence.md](campaign-evidence.md).
The scorer verifies both bindings, the producer fields and the runner bundle
digest. Failed or missing children cannot be promoted by constructing a
plausible-looking manifest from partial output. Per-leg supervision and
before/after process snapshots are retained even for failures.

Native ORT identity comes from the current process's loaded modules, after
ORT initialization, using [Process.Modules](https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.process.modules?view=net-10.0).
Its full mapped path, architecture and actual file hash are recorded; package
directory wildcard selection is not used. The SDK is embedded at runner
build time and matched to both core builds; the runtime patch is observed in
the running process. Linux host output includes package/core/SMT-sibling IDs.

`eng/campaign_processes.py` uses Linux `/proc/<pid>/stat` cumulative user/system
ticks and start ticks, or Windows CIM CPU counters and creation ticks. It
rejects malformed, missing, empty, non-finite, duplicate and backwards data,
checks the supervisor birth identity, and excludes only its descendant tree.
PID reuse cannot subtract unrelated old CPU or claim ownership through a
younger recycled parent. A new foreign PID contributes its full observed
lifetime CPU conservatively. More than 10% of guest-wide CPU capacity aborts
a release run. This module is separate from the historical PowerShell monitors.

Snapshots miss some CPU of short-lived/exited processes; the manifest reports
that limitation and the number of observed foreign disappearances. They do
not establish exclusive ownership of the machine or catch every transient
competitor. Isolation, the paired ORT controls and stability gates remain
necessary. Linux accounting/parser behavior has local fixture coverage;
record actual Linux collection/native-discovery evidence when validating a
new producer version on the assigned AMD VM. Windows collection and
loaded-module capture were exercised locally.

Local proof on September 17 used Windows/Intel, SDK 10.0.204 and runtime
10.0.12: all fifteen cases passed on each core and every cross-core input
digest matched. DINOv3's external tensor file was captured. The box had
roughly 16–21% observed foreign CPU, so no release/performance verdict was
claimed. A separate 33-sample E5-8 run exercised duration/convergence warmup;
limiting it to three warmup pairs produced the expected refusal. Kept evidence
is under `artifacts/amd03-common/`, including `local-validation.json`.
The final startup-affinity build additionally passed both E5-8 smoke legs
in `smoke-pinned-e5/`, with the CLR reporting one available CPU. The local
checks comprise 44 Python tests and 11 common-runner self-checks, including
actual Windows child affinity inheritance and parent restoration after a
failed launch. Linux launch wrapping has fixture coverage; live Linux
verification remains open.

For sample-level noise investigation, the ordinary Bench executable accepts
`--sample-diagnostics`. It emits one `sample-diagnostics` JSON record per case
after all timings and agreement checks. Rows preserve execution order, engine
and sample index, the unchanged stopwatch latency, surrounding observation
duration, thread/process CPU time, GC collection/pause deltas and current-thread
allocated bytes. Counter reads and record storage occur outside the headline
stopwatch; observation overhead and Windows CPU-counter granularity still make
these diagnostic runs. GC counters are process-wide and cannot alone attribute
a pause to an engine. Do not subtract counters to manufacture corrected latency
or filter inconvenient samples. The common campaign runner rejects this flag.

The managed core also has an off-by-default storage experiment:
`LOKAD_ONNX_FUSED_TEMP_RELEASE=1` enables the private MatMul-result release in
the trailing MatMul-plus-Div composite. The two arithmetic passes are unchanged;
only its unexposed intermediate is returned after Div, including on failure.
This experiment is off by default and does not reclaim graph intermediates.

`LOKAD_ONNX_SOFTMAX_EXP_PRUNE=1` uses the bitwise-equivalent exponential
experiment: lanes below the existing underflow cutoff use zero during polynomial
evaluation, then select the same positive-zero result as before. This avoids
computing discarded subnormal values. The cutoff, returned values, reduction
order and floating-point control settings are unchanged. Off by default.

`LOKAD_ONNX_RELEASED_BUFFER_CACHE=1` before process startup retains only arrays
already released by a completed call's lifetime checks. Each serialized graph
or explicit context has a separate cache, limited to 128 MiB of payload,
256 arrays and 32 arrays per type/length. Live outputs and intermediates are
never adopted by Reset. The pool's ownership/alias accounting and metrics are
fresh per call; cache transfers remain inside Execute. DisableBufferPool and
preparation invalidation clear retained storage. ExecuteNode keeps its existing
non-pooled path. Record this environment setting alongside binary identity in
diagnostics; it is not a promoted default or a qualified campaign result.
