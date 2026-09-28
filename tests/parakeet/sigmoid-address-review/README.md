# Resolve saved sigmoid sample addresses without another inference

The closed trace `c7382fcf` contains stack addresses and method rundown metadata.
The original all-event exporter retained rundown payloads without decoding their
names. Reading those bytes already identifies eight sigmoid method ranges,
including the 555-byte Tier1 helper and 3,240-byte Tier1 public operator.

Use the retained TraceEvent 3.1.23 library and its documented TraceLog conversion
to export every code-address entry, stack entry, sample and request boundary.
Run the reader with the already installed VM PowerShell on CPU0, using the frozen
export monitor. No product, consumer or analysis assembly is built; no model runs.
Raw sources are pinned under `artifacts/parakeet-sigmoid-address-review-20260928`:
[TraceLog v3.1.23](https://github.com/microsoft/perfview/blob/v3.1.23/src/TraceEvent/TraceLog.cs)
and the corresponding EventPipe and CLR parsers. Disable symbol downloads.

Require all 115,795 original samples and 120 request markers, zero lost events,
and unchanged input hashes. Reconcile every sample with the original exported
event by timestamp, process/thread, order and raw payload before attributing
addresses. Keep unknown frames, code versions and unmapped samples explicit.
Report sample counts separately from estimated thread time or instruction cycles.

Commands are `run.py stage`, `launch`, `observe`, then `collect`. Except observation,
run each lifecycle action once. Bounds remain 11 GiB available / 2 GiB tmpfs
before launch, 12 GiB RSS, 1 GiB remaining, 512 MiB output and 900 seconds.
Keep the derived ETLX on the VM with its recorded hash; collect at most 32 MiB
of compressed address tables and logs locally. All source evidence stays intact.
