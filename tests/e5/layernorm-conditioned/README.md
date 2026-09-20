# LayerNorm banks after fixed conditioning

The [completed result](results-20260920.md) passes every duplicate control and
all output/resource checks, but fails the fixed candidate regression screen:
the second worker's 8-token mean is 10.95% slower than Product. All other banks
pass their gain/regression checks. The [complete observations](observations-20260920.json)
retain all 13,824 measured batches, including the failing worker. Production
remains unchanged; no whole-model or native ORT improvement is established.

This distinct experiment retains every bank and threshold from the
[inconclusive prior result](../layernorm-bank/results-20260920.md). It changes
only measurement orchestration. Each bank first accumulates three seconds of
timed kernel work, stopping at the first complete four-role cycle to reach that
budget. Every conditioning record is retained. Then 96 measured cycles use
four times the prior repetitions: 512/128/32/32/8 for real e5 banks and 128 for
diagnostic banks. The four workers retain 13,824 measured batches and 144 first
calls. No observation is excluded or used to choose when measurement stops.

Product, the identical Copy A/B delegates and Wide retain the exact proven
arithmetic and all 125 real captures. Duplicate controls remain within 1%
overall and 2% per worker. Candidate gain must be at least 2% at 30/padded128/128
against all three controls; regressions may not exceed 1% overall or 2% per
worker on any of the nine banks. Passing this component screen allows product
integration and model qualification; it does not establish either by itself.

`prepare.py stage --artifact artifacts/e5-layernorm-conditioned-20260920`
verifies the preceding closed receipt, generates the host and tools from its
exact archived sources, and saves every source pin and the complete changes.
The generator makes only explicit, uniquely matched replacements. Unchanged
sources preserve their original bytes. Existing capture files are referenced
through the closed AMD proof, without copying them again.

Build `<artifact>/host/Probe.csproj` in Release with `--tl:off --nologo -v minimal`,
`FrozenProductDirectory=<absolute hardware artifact>/collected/product`,
`FrozenSourceDirectory=<absolute new artifact>/payload/source`, and output
`<artifact>/payload/bin`; retain `build.log`. Run generated
`<artifact>/tools/local_check.py --artifact <artifact>` for actual host identity
and independent checking of all nine input mappings, without kernel inference.
Run `<artifact>/tools/test_audit.py` and retain stdout/stderr as `unit-tests.log`.
Commit the generator, then run `prepare.py freeze --artifact <artifact>`.
Freezing verifies the generated and generator sources, build, local checks,
and every archive entry before recording the payload identity.

Use generated `tools/vm.py launch`, `poll`, then `collect`, each with
`--artifact <artifact>`. Launch requires both closed prerequisites, their actual
terminal process identities, matching remote captures and sufficient resources.
The four bounded sequential workers use AMD CPU 2, .NET 10.0.8, normal runtime
settings and no native ORT. Collection waits for every original process to end.

Run generated `tools/audit.py --payload <artifact>/collected --origin
artifacts/e5-layernorm-amd-proof-20260920/collected --output <artifact>/audit.json`,
then `tools/close.py --artifact <artifact>`. The closer separately recomputes
integer timing totals, every conditioning cutoff and all fixed decisions before
writing the report and receipt. Original unsuccessful controls stay preserved.
Each stage and successful writer is single use; retain failures without replaying
completed inference. No production source or default changes in this experiment.
