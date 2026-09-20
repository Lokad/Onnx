# AMD LayerNorm arithmetic and code qualification

This reuses the exact proof body and generated kernels from the
[closed local proof](../layernorm-output/README.md). That proof used software
Vector512 fallback. This experiment requires actual AMD AVX-512 and Linux
.NET 10.0.8, with the same 915 cases and 125 captured LayerNorm instances.

The clean proof permits no runtime overrides. A separate code phase explicitly
records two disassembly variables, repeats the full proof and warms complete
kernels for three seconds to observe optimized generated code. Neither phase
measures a speedup. Actual output bits, source identities and code inspection
must pass before a separate timing experiment is justified.

Run `prepare.py stage --artifact <fresh-directory>`, then build `Probe.csproj`
with `--tl:off --nologo -v minimal`, `FrozenProductDirectory=<artifact>/payload/product`,
`FrozenSourceDirectory=<artifact>/payload/source` and output `<artifact>/payload/bin`.
Run `local_check.py --artifact <directory>` for the identity-only host check and
`test_audit.py`, retaining `build.log` and `unit-tests.log`. Then
`prepare.py freeze --artifact <directory>` creates the pinned portable payload.

`vm.py launch --artifact <directory>` requires the entire independent e5
deployment campaign's closed receipt and verifies its process termination.
It checks 8 GiB available memory and 512 MiB free disk before transfer. Do not
launch it while the primary campaign is active. Poll the original deployment
identity, then collect after every observed birth is terminal. Collection streams
the archive locally, avoiding an extra remote archive copy.

Run `audit.py --payload <artifact>/collected --output <artifact>/audit.json`.
Inspect the actual optimized disassembly and record `code-review.json` with
`passed`, the `jit` file pin, the auditor's `code` object and concrete observations.
Then run `close.py --artifact <directory>` and independently reverify its inventory.
Successful writers are single-use. Preserve failed runs and transport attempts;
an observation timeout never licenses restarting inference.
