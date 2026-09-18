# Isolated e5 evidence, prospective schema 3

This protocol runs one engine and one canonical e5 case per fresh process.
It addresses measured interference from managed background GC during native ORT
calls in the older paired process. Schema 1/2 evidence keeps its original meaning;
it cannot calibrate schema 3. No samples, GC pauses or outliers are removed.

The common Campaign executable still compiles once against the released public
API. Only the core DLL changes between L0 and L1. `isolate oracle|lok|ort config.json`
is its new producer entry point. Configuration, process JSON, stdout/stderr,
supervision and before/after process snapshots are bound to SHA-256. Models are
local; no downloads, symlinks or mutable shared output directories are used.
The seven Python producer/validator sources are copied into each evidence tree,
hashed, rechecked after collection and included in A/A/comparison identity.

The five cases, in order, are e5-8tok, e5-30tok, e5-30pad128, e5-128tok and
e5-512tok. The first four are primary. Inputs come from the existing canonical
`Bench.E5Inputs`; the padded case has thirty real tokens in a 128-token tensor.
This version pins the inspected e5 model (SHA-256
`ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`), whose weights
are embedded. Tokenizer and all input bytes/dtypes/dimensions are also hashed.

An untimed native oracle process produces all named float32 outputs, with exact
dimensions and per-file digests. Each worker independently rebuilds canonical
inputs and checks the oracle manifest digest, asset identities and every output
before and after timing. Scaled error remains <=1e-4, with finite values and
unchanged inputs required. Native workers check the actual buffers given to ORT.
The Lokad worker must have no native ORT library loaded, before or after running.
The native worker must load the oracle's exact native library. Oracle output
digests, native identity and workload identities must match A/A and comparison.

The supervisor confines each child to one logical CPU before runtime startup.
Workers verify the affinity and the CLR's one-CPU view. Managed execution uses
cached `ExecutionOptions.Default` (automatic SIMD/FMA, degree one); ORT uses
CPU-only, all graph optimizations, sequential execution, one intra/inter-op
thread, and no spinning. Runtime, SDK, actual GC mode and inherited
DOTNET/COMPlus/LOKAD variables are recorded and held constant. Default tiering and
full optimization require separate calibrations. Neither GC mode nor numeric
precision is changed by this protocol.

Each worker warms its own engine for at least 1000 ms of measured calls and until
the last nine calls' range is <=10% of their median. All warmup ticks are retained.
At most 1000 calls/60 seconds are allowed. Failure aborts the campaign and preserves
partial evidence; it never yields a scored subset. The subsequent **33 individual
public Execute/Run stopwatch values** remain the primary metric. Reset, disposal,
output reads, hashes and observation counters are outside those stopwatches.

The 33 calls are also grouped into three blocks of eleven. Each block stopwatch
includes Reset, disposal and raw-tick recording; its CPU, allocation and GC
counters are observed outside the block. These complete-request measurements are
separate diagnostics. They never replace or adjust individual Execute/Run samples.
First-call/load observations also remain separate.

There are four paired repetitions in the existing order:
L0,L1 / L1,L0 / L1,L0 / L0,L1. Each role visit contains ten fresh workers: the five
cases, two engines each. Engine order is lok,ort on even zero-based visits and
ort,lok on odd visits, balancing both core roles. The default scored cooldown is
300 seconds between pairs. Every child is sequential and supervised; overlapping,
missing, failed, reused or reordered workers invalidate the evidence.

The acceptance policy is unchanged: raw range <=25% of median, half-split drift
<=10%, A/A and comparison process-median/ratio variation <=3%, regression limit
max(3%, twice A/A variation), and candidate/ORT <=1.05 on each primary case.
Native controls must remain stable across the paired roles. All five cases must
complete in this initial isolated version. A/A must finish and satisfy its gates
before any scored comparison starts. Mixed schema, runner, runtime, host,
settings, assets or native binary identities are rejected.

Smoke uses one warmup and two timed calls and is always marked unscored. The
worker can be tested independently before the supervisor/scorer is ready:

```powershell
dotnet build tests/Lokad.Onnx.Campaign/Lokad.Onnx.Campaign.csproj -c Release --tl:off --nologo -v minimal -p:CoreAssemblyPath=C:/absolute/path/to/released/Lokad.Onnx.dll
dotnet tests/Lokad.Onnx.Campaign/bin/Release/net10.0/Lokad.Onnx.Campaign.dll selftest
python eng/test_isolated_worker.py --runner tests/Lokad.Onnx.Campaign/bin/Release/net10.0/Lokad.Onnx.Campaign.dll --output artifacts/isolated-smoke-new --source 4495fc68b9505b0b6fab73146bd905424588218e --cpu 4
```

The smoke retains full numerical/identity checks and deliberate refusal probes.
Choose an allowed local CPU and a new output directory. A smoke pass establishes
producer behavior, never performance. No historical experiment is reclassified
as schema-3 evidence.

After preparation, the supervised commands are:

```powershell
python eng/test_isolated_campaign.py
python eng/run_isolated_e5.py --prepared artifacts/common-prepared-new --output artifacts/isolated-smoke-new --cpu 2 --kind comparison --jit full-opts --smoke
python eng/run_isolated_e5.py --prepared artifacts/common-prepared-new --output artifacts/isolated-aa-new --cpu 2 --kind aa --jit full-opts
python eng/run_isolated_e5.py --prepared artifacts/common-prepared-new --output artifacts/isolated-comparison-new --cpu 2 --kind comparison --jit full-opts --aa artifacts/isolated-aa-new/evidence.json
```

Use `--root /absolute/repository` when the scripts live in an immutable artifact
directory and models live in the repository. This reads the existing assets.
Keep prepared binaries and producer sources identical across both runs. Set any
experimental core switches in the child-inherited environment before **both**
calibration and comparison; their values are part of the captured identity.
Default tiering is a separate run without `--jit full-opts`. The supervisor
rejects an unqualified calibration before staging a comparison. The offline
scorer remains `python eng/score_campaign.py --evidence ... --aa ...`.
