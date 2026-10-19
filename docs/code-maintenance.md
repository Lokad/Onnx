# Source and supporting-tool maintenance

The core library is packaged; Data, CLI, tests and maintenance tools are not.
Public library APIs can have external callers. A lack of repository references
does not authorize deleting them.

## Source audit and TODO triage

Run from the repository root after restoring the qualified tokenizer dependency:

```powershell
dotnet build eng/code-audit/CodeAudit.csproj -c Release --tl:off --nologo -v minimal
dotnet eng/code-audit/bin/Release/net10.0/CodeAudit.dll --self-test
dotnet eng/code-audit/bin/Release/net10.0/CodeAudit.dll --repo . --check --output artifacts/code-audit
```

The nonpackaged tool uses the selected SDK's Roslyn compiler analysis assemblies.
It loads every project in `Lokad.Onnx.slnx` and rejects incomplete loads or C#
compilation errors. Generated declarations are excluded; generated callers still
count. Its report distinguishes public APIs, production uses, test/benchmark
uses, language/framework contracts and registered reflection consumers. It
examines source references, rather than proving complete runtime reachability.

Review a new private/internal finding before removing it. Intentional test
diagnostics and reflection consumers have individually justified entries in
[retained-members.json](../eng/code-audit/retained-members.json), keyed by assembly
name plus compiler documentation member ID. Missing members, unnecessary
exceptions and lost registered consumers fail the check. Reflection strings in
standalone tools require explicit registration; the diarization detail tool's
`WeSpeakerEmbedder.ExtractPipeline` call is one example. No exception list is
generated or accepted automatically.

The fixtures exercise unused helpers, external APIs, entrypoints, constructors,
disposal, ordinary and static-abstract interface implementations, test-only
references, reflection registrations, stale entries and failed compilations.
CI runs this bounded check once in the ordinary Ubuntu test job. It does not
scan historical Python scripts or add compiler dependencies to the package.

[TodoTriageTests](../tests/Lokad.Onnx.Tensors.Tests/TodoTriageTests.cs) separately
checks TODO comments. Its existing Span.IndexOf and MOVBE triage remains in
place; it is not an unused-code detector.

## Active entrypoints outside the main solution

| Entrypoint | Role and constraint |
|---|---|
| [Lokad.Onnx.Campaign](../tests/Lokad.Onnx.Campaign/Lokad.Onnx.Campaign.csproj) | Shared graph campaign runner compiled against an explicit core DLL. Its historical dependency pins are part of that protocol. |
| [Lokad.Onnx.EmbeddingGemma2Bench](../tests/Lokad.Onnx.EmbeddingGemma2Bench/Lokad.Onnx.EmbeddingGemma2Bench.csproj) | Owned-vector FP32 benchmark with ORT 1.29.0; one unchanged runner is staged with each measured core. |
| [Diarization detail trace](../tests/pyannote/diarization/Detail.cs) | Intermediate speaker-pipeline comparisons, including registered reflection calls. |
| [Common campaign builder](../eng/build-common-campaign.ps1) and [Gemma builder](../eng/build-embeddinggemma2-benchmark.ps1) | Build pinned sources from git archives into fresh directories without extra worktrees. |
| [Common supervisor](../eng/run_common_campaign.py) and [Gemma supervisor](../eng/run_embeddinggemma2.py) | Process ownership, accounting, source/input identities and preserved timing samples. Inspect protocol requirements before reuse. |

Correctness lanes remain `eng/test-opfuzz.ps1`, `eng/test-e5.ps1` and
`eng/test-embeddinggemma2.ps1`. Shared reference kernels under
`tests/Shared/ReferenceKernels/` compile only into tests and benchmark tools;
they retain independent arithmetic and must not be replaced with calls to the
optimized implementation they verify.

## Intentional numerical specializations

Production references retain these methods without unused-member exceptions.
Their stable identities and reasons matter when interpreting clone reports:

| Member identities within assembly `Lokad.Onnx` | Reason for separate compiled entrypoints |
|---|---|
| `M:Lokad.Onnx.MathOps.ErfVector(System.Numerics.Vector{System.Single})` and `M:Lokad.Onnx.MathOps.ErfVectorInline(System.Numerics.Vector{System.Single})` | Equal arithmetic with deliberately different inlining behavior. |
| `M:Lokad.Onnx.MathOps.ExpVectorEstrinPruned(System.Numerics.Vector{System.Single})` and `M:Lokad.Onnx.MathOps.ExpVectorEstrinPrunedInline(System.Numerics.Vector{System.Single})` | The inline route preserves vectors across polynomial evaluation. |
| `M:Lokad.Onnx.MathOps.ShortWideMultiply3Rows(System.Int32,System.Int32,System.Int32,System.Single*,System.Single*,System.Single*)` and the shared three-row kernel in `MathOps.cs` | Shape-specific dispatch and compilation attributes were qualified together; equal bodies do not establish equal generated code. |

The packing/remainder helpers in `Zzz.IsolatedShortMatMul.cs` and AVX2/AVX-512
convolution and Winograd kernels have the same need for separate qualification.
Changing these families requires output/exceptional-value checks and affected
model timings. Source generation, if ever useful, must preserve distinct
compiled methods and their attributes.

## Frozen and unclassified experiments

The forty C# paths in
[closed-experiment-sources-20260928.json](../tests/Shared/closed-experiment-sources-20260928.json)
have enforced path/content identities. Do not edit, move or deduplicate those
inputs. Other closed experiment producers and their receipts follow the
[artifact retention policy](artifact-retention.md). Being outside the solution
alone does not classify a tool as dead or frozen: identify its runner, consumers
and evidence before changing it.

One bounded follow-up is to provide a shared launch/observe helper for the next
e5 diagnostic runner, using the repeated orchestration in
`tests/benchmarks/e5-direct-code-diagnostic-amd/run.py` and
`tests/benchmarks/e5-direct-tier-diagnostic-amd/run.py` as reference inputs.
Pass their module-global dependencies explicitly into the shared process helper
and qualify command/environment/timeout handling with fixtures. Leave the old
producer files and evidence unchanged; this does not authorize a historical
script migration.
