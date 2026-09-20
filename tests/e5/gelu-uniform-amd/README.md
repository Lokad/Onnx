# AMD conditional GELU comparison

This experiment measures the exact all-small shortcut from the
[local arithmetic proof](../gelu-uniform-shortcut/README.md), on the complete
twelve-layer banks from the [e5 census](../gelu-branch-census/results-20260920.md).
It changes no product code and measures no ORT or complete-model latency.

The payload explicitly contains the 120 required input/bias arrays and their
original closed receipt. It is a verified subset, not a copy of all captured
outputs. The host validates exact subset coverage and original file identities.
The unchanged `Kernels.cs` has two duplicate controls and one conditional body;
the actual product delegate is a fourth variant. Each worker performs the full
1,575-case bit proof before any timing. Expected complete-bank hashes also bind
the independently retained census outputs.

An initial AMD code worker captures the optimized bodies. Timing starts only
after recording exactness and an actual branch bypassing the unused large-value
polynomial/exponential. Timing workers use normal runtime settings, CPU2 before
startup, with the supervisor on CPU0.

Four fresh workers rotate/reverse the five-case order. Each case uses sixteen
warmup cycles and forty-eight measured cycles, with every variant once per cycle
in rotating order. Each sample includes 64/32/8/8/2 complete twelve-layer banks
for 8/30/padded128/128/512 tokens. All 3,840 measured batches and 1,280 warmup
batches remain, with allocations, GC, output checks and actual resources.

The fixed engineering screen requires aggregate duplicate means within 1%, each
worker within 2%; at least 2% candidate improvement against every control at
30/padded128/128; no aggregate regression above 1% or worker regression above 2%
on any case. Failed controls make the timing inconclusive. These limits do not
provide calibrated confidence or qualify a whole-model improvement.

`prepare.py` generates a new host from the closed proof source and copies the
verified input subset. Build its generated project against the frozen qualified
core with `--tl:off --nologo -v minimal`. `remote.py code <artifact>` runs the
initial code proof. After its separately recorded code gate, `remote.py launch`
starts the fixed four-worker schedule; `collect` requires terminal births and
revalidates all payload files. Collection includes every new result/source file;
the already verified data/binaries remain in the local payload.

`audit.py --artifact <collected> --payload <local-payload> --output <new-audit.json>`
checks every record and reports the unchanged screen. `test_audit.py` exercises
coverage/order/output/count refusals and failed-control/failed-case verdicts.
Successful preparation, execution, collection and audit writers are single-use.
