# Managed Whisper GELU diagnostic

This compares the qualified managed core on the existing 41-output encoder trace
and the same graph with all 34 Erf outputs and 32 final feed-forward activation
outputs exposed. These prevent GELU and unintended ScaledMatMul fusions while
preserving every original operation and constant.
The actual optimized graph comparison rejects unrelated optimizer changes.
This is a numerical diagnostic; additional output lifetimes prevent latency claims.

Run from the repository root. Build the standalone consumer with
`dotnet build tests/whisper/managed-gelu-v2/Probe.csproj -c Release --tl:off --nologo -v minimal`,
passing `-p:FrozenProductDirectory=<absolute frozen product directory>` and
`-o artifacts/whisper-managed-gelu-v2-20260921/bin`. Preserve the build log in
the artifact directory. The expected product directory is recorded in
`experiment.py`; the core SHA256 is checked explicitly.

Use `C:/Python313/python.exe -X utf8 -B` for `experiment.py prepare`, then
`run.py`, then `analyze.py` in this directory. Preparation binds existing model,
input, older managed trace, and two independent double-reference files. There
are seven fresh calls plus one verified reused baseline: baseline and unfused
GELU on three clips plus a repeat. The first design's baseline remains valid;
its failed variant and subsequent memory preflight refusal are retained separately.
Do not rerun a completed campaign or overwrite the manifest. The supervisor
retains failures and terminates only its identified worker on a resource breach.

The prospective mechanism screen requires at least halving final maximum scaled
error on every unique clip against both references. The full-output `1e-4`
numerical gate stays unchanged. Every common boundary and padding element is
included. This small diagnostic cannot qualify the full audio application.
