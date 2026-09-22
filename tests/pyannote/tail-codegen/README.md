# AMD narrow-convolution code generation

This diagnostic loads the exact selected Core and two rejected probe assemblies.
It binds delegates to their existing private `Data`, `Iterations` and `Validate`
methods. The kernels and internal workload call sites are not rebuilt. The
outer driver declares `DOTNET_JitDisasm` and leaves tiering and ISA flags alone.

The original 22-shape conditioning order, further warmup and six blocks per
shape remain. Output hashes replace timing scores. Full original validation runs
afterward. This is a fresh diagnostic process, so its dynamic profiling state
must not be assumed identical to the historical timing processes.

From the repository root, run `C:/Python313/python.exe -X utf8 -B` with
`tests/pyannote/tail-codegen/run.py prepare`, then separately `stage`, `launch`,
`observe`, and `collect`. Collection requires authoritative terminal identities.
Finally run `tests/pyannote/tail-codegen/audit.py` with the same Python command.
All writers refuse existing completed artifacts; do not repeat a completed stage.

Artifacts use `artifacts/pyannote-tail-codegen-20260922`. Raw logs retain all
compilation tiers and any interleaving. Extracted snippets are explanatory only;
the audit requires a complete, uninterleaved optimized Tier1 listing for each
selected method. Resource checks, full-buffer hashes, exact probe/Core identities
and installed runtime pins remain mandatory. No timing ratio or product selection
follows from instruction inspection alone.

The diagnostic flag is documented in the [.NET runtime repository](https://github.com/dotnet/runtime/blob/main/docs/design/coreclr/jit/viewing-jit-dumps.md).
