# AMD E5 campaign evidence, schema 1

Run the offline scorer with a comparison manifest and a preceding unchanged
A/A manifest:

```powershell
python eng/score_campaign.py --evidence artifacts/candidate/evidence.json --aa artifacts/unchanged/evidence.json
python eng/test_campaign_scorer.py
```

The scorer never connects to a benchmark machine, loads a model, or changes
an input file. It checks raw logs and the observations recorded by the runner.
The manifest is an evidence index, not cryptographic attestation of a machine.
A trusted producer must capture the fields below from the processes actually
measured. Do not fill absent observations by guessing from a historical log.

The old log-only scoring invocation deliberately returns ABORT (exit 2).
`eng/parse_baseline.py` still provides historical log-only diagnostic tables.
Use the new [common runner and supervisor](common-campaign.md) to produce
this contract. The older `eng/run-l0l1-canonical.ps1` still uses the retired
invocation and two different runners; its five-case baseline is insufficient.

## Measurements and policy

Both campaigns contain four pairs of fresh processes, 33 or more samples per
case/engine/process, in role order `L0,L1 / L1,L0 / L1,L0 / L0,L1`. The complete
case order is:

```text
e5-8tok e5-30tok e5-30pad128 e5-128tok e5-512tok
dinov3-224 resnet50-224
gpt2-1tok gpt2-4tok gpt2-32tok gpt2-128tok
gpt2-dec-p1 gpt2-dec-p32 gpt2-dec-p128 gpt2-dec-p512
```

Every case must report successful numerical agreement before/after timings
and intact inputs. Context timings remain diagnostic; public Execute (`lok`)
and ORT Run (`ort`) are the scored boundaries. Keep each raw series in sample
order. The range divided by median must be at most 25%; the difference between
chronological half medians divided by the full median must be at most 10%.

For each case, derive unchanged-run variation from the A/A campaign alone:
take range/median of all eight process medians for Lokad, ORT, and their
within-process Lokad/ORT ratios, then use the largest of those three values.
The regression allowance is `max(0.03, 2 * variation)`. A/A variation above
3% is INCONCLUSIVE, as is variation above 3% among the four process medians
or ratios within either comparison arm. These 3% caps are provisional,
explicit policy constants, frozen before an experiment; noisy candidate
measurements cannot enlarge their own regression allowance.

The two ORT controls remain separate. Each pair's ORT change must fit
`max(0.03, 2 * A/A ORT variation)` or the comparison is INCONCLUSIVE. Each
output row includes the four per-repetition Lokad and ORT medians and ratios.
`R0` and `R1` are medians of within-process Lokad/ORT ratios; `O0` and `O1`
are separate medians. Improvement uses the median of the four paired
`L1/L0` ratios, not a pooled ORT denominator. Both raw and ORT-normalized
regressions are checked against the frozen allowance.

PASS means each primary case (`e5-8tok`, `e5-30tok`, `e5-128tok`,
`e5-30pad128`) has `R1 <= 1.05`, with no regression in any required case.
Otherwise a steady valid comparison is MISS, or REGRESSION if a regression
exists. The E5 equal-weight geometric improvement is descriptive. There is
no cross-model family target or required ResNet improvement. PASS is a
parity measurement objective, not proof of a new gain or permission to ship.
Optimization promotion still needs a gain outside unchanged-run variation,
memory bounds and correctness/distribution checks.

Exit codes are 0 for scored PASS/MISS/REGRESSION, 2 for invalid/missing evidence,
3 for INCONCLUSIVE measurements, and 4 for no scoring arguments. Read the
JSON verdict rather than interpreting exit 0 as release acceptance.

## Manifest

The root object has `schema: 1`, `kind: "aa"` or `"comparison"`, and a `runs`
array containing eight records. Unknown extra fields may retain diagnostics;
duplicate JSON keys and non-finite JSON numbers are errors. Each record has:

| Field | Meaning |
|---|---|
| `role`, `rep` | `L0`/`L1`, and integer 1 through 4, in the order above. |
| `log`, `log_sha256` | Relative path from the manifest directory (or absolute path), and SHA-256 of the original complete log bytes. Paths must be distinct. |
| `process_id`, `started_utc`, `completed_utc` | Actual fresh process identity and observed launch/exit timestamps with UTC offset. Positive PID, positive duration, no overlapping legs. A/A must finish before comparison starts. |

`started_utc` is the runner-observed wall clock at managed entry, deliberately not the kernel birth tick: on Linux the birth tick truncates to the HZ boundary (up to ~10 ms early) and can precede the supervisor wall mark, falsely aborting live runs (first seen on the AMD quiet box, September 17). Completion stays wall-clock; PID identity still binds the process.
| `exit_code` | Actual process exit code, exactly zero. |
| `source_sha`, `core_sha256` | Full 40-hex git commit and SHA-256 of the core assembly actually used. Stable within each arm. Both A/A arms and comparison L0 must use the same source and binary. |
| `runner_sha256` | SHA-256 identifying the identical common workload runner used in every process, including A/A. |
| `ort_native` | Object with `path`, `sha256`, `architecture` for the **actually loaded** native ORT runtime module. |
| `environment` | Process settings described below; identical across both campaigns. |
| `accounting` | Object with `valid: true` and numeric `foreign_cpu_fraction` from valid process accounting; fraction must be in [0, 0.10]. |
| `cases` | Object keyed by all fifteen names, each with full `model_sha256`, `input_sha256` and explicit `external_data` map; E5 additionally has integer `unmasked_tokens`. |
| `cases_failed` | Quarantined case names in canonical order (identities present, no timed rows). `cases` covers all fifteen attempted names; `cases_failed` is the subset without rows. |

The common producer also emits `producer: "common-runner-v1"`, `runner_files`,
`process_evidence` and `process_evidence_sha256`. The referenced child JSON is
bound to the manifest; its observed identity/settings/cases/cases_failed must match. Only
the completion timestamp may be extended to the supervisor's observed exit.

A quarantined case contributes no timed rows: its log carries the case header, casedef and
warmup series with a `FAILED` status but no summary or raw series, and the scorer reports it
INCONCLUSIVE (never a silent pass). A quarantined primary (E5) case makes the campaign
INCONCLUSIVE. First instance: resnet50-224 warmup never converged (strict period-2 fast/slow
alternation on both engines, prime suspect GC cadence under workstation GC); the leg's good rows
remain usable instead of aborting the campaign.
`runner_sha256` for this producer hashes UTF-8 concatenation of ordinally
sorted `filename + NUL + lowercase file SHA-256 + LF` entries in `runner_files`.
The bundle contains top-level managed DLLs, deps.json and runtimeconfig.json,
excluding only Lokad.Onnx.dll. Thus changing a tokenizer/runtime dependency or
runner configuration requires fresh matching A/A, as well as changing code.

All SHA-256 digests have 64 hex characters. Hash the loaded native module,
not an arbitrary package asset: record an absolute path ending in `onnxruntime.dll` or
`libonnxruntime.so` (optionally with numeric suffix). Its process architecture
must be `x64` or `arm64` and match the environment. If its path contains a
`runtimes/<rid>/native` directory, that RID must also match the architecture.
Both arms and A/A use the same native digest; installation paths may differ.
The scorer checks declarations and log bindings; the producer is responsible
for observing the loaded module rather than selecting a package wildcard.

`environment` contains exactly `host`, `cpu`, `os`, `architecture`, `sdk`,
`runtime`, `isa`, `affinity`, and `settings`. All except `settings` are nonempty
strings. Host, CPU identity, actual runtime patch (e.g. `.NET 10.0.8`), ISA and
affinity must agree with the log. `settings` contains a nonempty `gc` string
describing the actual GC configuration, `jit` (`default-tiered` or `full-opts`),
and `variables`, a dictionary of all effective DOTNET/COMPlus/LOKAD environment
overrides relevant to execution. Values are strings; omit unset variables.
Capture these in the measured process, including inherited overrides.
`full-opts` requires an explicit tiered-compilation value of `0`; conflicting
DOTNET/COMPlus tiering aliases are rejected. JIT regimes must match the A/A
campaign. Full-opts evidence is explicitly labelled in the output and must
not be presented as default-tiered deployment performance.

Input identity is SHA-256 over all named input tensors, including masks,
padding, token types and past tensors. The common producer writes ASCII
`LOKAD-CAMPAIGN-INPUTS-1` followed by NUL, then a little-endian int32 tensor
count. Tensors are ordered by ordinal name. Each record contains int32 UTF-8
name length and name bytes, int32 ONNX dtype (7=int64, 1=float32), int32 rank,
int32 dimensions, int64 element count, then little-endian element bits in
logical tensor order. Float signed zero/NaN bit patterns are not normalized.
Unsupported input dtypes fail. Do not hash only text or shapes.

`model_sha256` hashes the full ONNX file, preserving its casedef sha12 prefix.
`external_data` separately maps relative tensor-file locations to full SHA-256
of each referenced file; it must be explicitly `{}` for embedded-only models.
These maps participate in cross-run/A/A identity checks. **DINOv3 has an
external model.onnx_data file**; the earlier claim that all assets were
self-contained was incorrect. Do not fold external hashes into model_sha256
and thereby break the ONNX-file prefix check. Any
tokenizer/preprocessing change that changes tensors changes the input digest.
For E5 also record the measured count of nonzero attention-mask elements:
8, 30, 30, 128, 512 respectively in canonical order. In particular, the padded
case has shape `1x128` with 30 unmasked elements. Never label fully unmasked
128 tokens as the padded workload. Model/input identities must match every
process across both campaigns. Numerical outputs need agreement within the
existing tolerance, not byte-identical output hashes.

GPT-2 decode past comes from the common ORT prefill reference in the campaign
runner. Deriving it from each Lokad core's outputs could change the measured
inputs between baseline and candidate even when both pass numerical tolerance.

## Required log records and runner work

Each log contains exactly one `host=` line and one `confinement wallMs=` line,
and exactly one `case`, `casedef`, `warmup`, raw `lok/ctx/ort`, and `case-status`
record for every case. This is the current canonical Bench text syntax.
Duplicate, missing and extra cases are errors. UTF-8 and BOM-marked UTF-16
logs are supported. Runtime and CPU fields retain their full multiword values.

Host fields must show one affinity bit, one engine thread, CPU-only ORT,
intra-op=1, inter-op=1, sequential mode, optimizations ALL and no spinning.
The approximately two-second confinement check must have CPU/wall in
[0.8, 1.3], with internally consistent counters. The casedef model prefix,
iterations, warmup policy and input/output shapes must match the manifest
and other processes. Timings must be finite and positive and have exactly
the declared sample count. Summed sequential timings cannot exceed the recorded
process lifetime (allowing for printed rounding). `maxScaled` and `postScaled` must pass the declared
tolerance, which cannot exceed 1e-4; `inputsIntact=yes` is required.

Warmup must record at least three iterations and **at least 1000 ms of
measured execution per engine per case**. This duration is a provisional
minimum, not a claim that one second proves JIT steady state. Fixed warmup
must match its declared count; adaptive warmup must report `stop=steady`
within the declared bounds, never `max-reached`. The same policy applies to
A/A and comparison. The common producer now implements the duration floor,
a stable last-nine-sample window, actual identity capture and supervision.
Its local smoke proof does not establish AMD JIT convergence or provide
unchanged AMD calibration. Historical logs remain historical.

The test suite constructs complete synthetic manifests and canonical logs,
then mutates individual observations to prove rejection and scoring behavior.
Those fixtures contain invented digests and timings, are confined to temporary
directories, and are never AMD performance evidence.
