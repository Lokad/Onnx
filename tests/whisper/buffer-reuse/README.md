# Whisper bounded reuse prototype

This private source experiment retains encoder and decoder execution contexts
across serialized public requests. Its encoder can retain up to 512 MiB of
already-released arrays; decoder budgets remain 128 MiB each. The ordinary core
factory and the production source tree are unchanged. No forced collection or
runtime override is added.

`prepare.py` archives revision `18e10e3`, applies three recorded patches and runs
backend contracts. Its first archive omitted the CLI prerequisite: four CLI tests
failed, while 3,085 passed and 93 were skipped. `complete_build.py` supplies the CLI
from that same revision and retains both test attempts. The complete suite then
passes 3,089 tests with 93 skips. `prepare_consumer.py` builds the original public
Whisper consumer with read-only memory and pool telemetry.

The fixed AMD schedule is 20 conformance requests, followed conditionally by 80
requests in a separate process. All original application, input, held-output,
repeat and resource checks apply. Before endurance, encoder fresh pool payload
after the first request must be at most 16 MiB, and allocation across requests
2–16 must be at most half of the original failed normal-runtime prefix. Cache
limits and the warm encoder allocation limit apply throughout endurance too.
Raw elapsed times are diagnostic fields, not a matched ORT benchmark.

Run tools from the repository root with `C:/Python313/python.exe -X utf8 -B`.
Preparation and deployment create new evidence and refuse existing targets.
Do not rerun a launched worker. `collect.py` requires all actual process births
to be terminal and collects failures as well as successes. The artifact is
`artifacts/whisper-buffer-reuse-20260920` on both local and AMD workspaces.
The experiment is [closed and independently verified](results-20260920.md): all
100 requests and resource gates pass, and matching warm public allocations fall
75.53%. This justifies broader production qualification; it does not resolve the
separately recorded encoder/logit numerical differences or supply an ORT timing.
