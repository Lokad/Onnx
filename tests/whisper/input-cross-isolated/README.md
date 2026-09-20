# Full Whisper encoder comparison in finite processes

Preserve all twenty recordings plus the first repeat, all four input/engine
combinations and all 42 exact baseline bridges. Each of 42 sequential workers
loads one encoder and performs its baseline call followed by its crossed-input
call, then exits. This bounds process lifetime without raising the 8 GiB RSS
guard or forcing garbage collection. The two earlier attempts remain closed
resource failures.

Managed workers use fresh Memory contexts for each call. Hold each baseline's
actual output through the crossed call and Reset. Native workers similarly hold
their baseline output through the second call. Repeat equality is checked across
the fresh processes for requests 0 and 20. This is full-corpus numerical
localization, not a long-lived API memory or performance qualification.

Build `Probe.csproj` with the original `FrozenProductDirectory`, .NET terminal
logger disabled, into a fresh artifact's `bin` directory. Run `test_audit.py`,
`prepare.py --artifact <directory>`, then `run.py --artifact <directory>`.
The coordinator pins CPU affinity, records process births and resource samples,
and waits for each worker's actual termination before starting the next one.
Insufficient preflight memory waits at most 900 seconds; no worker is retried.

After the coordinator and all workers are terminal, run
`audit.py --artifact <directory> --output <directory>/audit.json` and
`close.py --artifact <directory>`. The auditor reads every full array and checks
all identities, baseline/input/held/repeat contracts, sequence, nonoverlap,
resource samples and numerical decompositions. Numerical failures remain failures.
All successful destinations are single-use; retain incomplete attempts.
