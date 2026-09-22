# Selected Pyannote attribution on AMD

The [completed AMD results](results-20260922.md) preserve all 48 public
requests exactly. Two reconciled captures put the packed three-row kernel at
53.33–54.17% of full-request sampled thread time. All owned processes are
terminal; preparation, collection, exports and independent audit have completed.
Do not relaunch this closed experiment.

This diagnostic keeps the exact selected Core `e9c87932` / Data `85d166b5`
product bytes. It adapts only the existing diagnostic consumer's thread-ID
lookup for Linux. Compiled inspection requires one changed existing method,
two new platform imports and unchanged public declarations. All 157 other
existing methods remain identical.

The original preparation used the wrong root for manifest-relative assets and
stopped before builds. The second built successfully, but adding imports above
an existing method renumbered compiler-generated methods and failed the strict
instruction check. Both preparations are preserved. The third places the new
imports after existing methods and passes the original instruction gate.
These corrections do not change product or public-result validation.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root with:

    tests/pyannote/amd-profile/successor_v3.py prepare
    tests/pyannote/amd-profile/successor_v3.py stage
    tests/pyannote/amd-profile/successor_v3.py launch
    tests/pyannote/amd-profile/successor_v3.py observe
    tests/pyannote/amd-profile/successor_v3.py collect
    tests/pyannote/amd-profile/successor_v3.py export
    tests/pyannote/amd-profile/successor_v3.py audit

Preparation, staging and launch each refuse existing destinations. Do not run
them again to poll an existing job. `observe` checks actual PID/birth identities;
an SSH timeout is not proof of termination. `collect` requires every owned
process terminal and streams evidence directly to the workstation.

Artifacts are `artifacts/pyannote-amd-profile-v3-20260922` and remote
`/dev/shm/lokad-pyannote-amd-profile-v3-20260922`. The root disk is nearly full;
payloads, diagnostics and temporary files use this bounded tmpfs directory.
No product build occurs on the VM. Existing .NET 10.0.8, Python, psutil and
model identities are verified against frozen inventories.

One control and two sampled processes run sequentially. Each executes all four
original fixtures with one warmup and three measured passes: 48 requests,
36 measured. The collector attaches after warmup and must enable the original
markers before the explicit release barrier opens. Every request retains native
speaker, timeline, centroid, input, ownership and repeat checks and must match
the earlier selected AMD output exactly.

Target affinity is CPU2 before CLR; collector and monitor use CPU0. Every
observed native thread must have its role's affinity. Each pair has 900-second
and 8 GiB RSS bounds; preflight requires 10 GiB available, staging requires
3 GiB tmpfs free, and execution preserves 1 GiB available and tmpfs free.
Output is bounded by 1 GiB per process pair and 2 GiB for the experiment.
The release barrier has a 180-second deadline. Exact PID/start-time ownership
remains mandatory; the consumer's separate wall-clock birth metadata allows
1.1 seconds for Linux boot-time rounding.

The pinned dotnet-trace 10.0.745401 exports each immutable trace to Speedscope
and Chromium locally. The original parser reconciles every exported event and
every fixture's complete request coverage. Sampled managed thread weights and
measured process CPU remain distinct. This diagnostic supplies no new ORT
timing ratio or candidate speed selection.

`report.py` initially assumed one exclusive bucket per method and stopped
before writing outputs: TransposeInto also has a zero-weight managed bucket.
The additive `report_v2.py` sums and retains every matching bucket. The failure
receipt is in `artifacts/pyannote-amd-profile-report-20260922`; original capture
and audit files are unchanged. The report successor completed successfully.
