# Prospective local complete-application packing trial

Run only after both higher-budget configurations pass the complete trajectory,
native-regression and memory audit. Pyannote's AMD lane retains priority, and
the live e5 campaign is untouched. This Windows trial can select an AMD follow-up;
it does not establish calibrated parity or authorize product promotion.

Four roles run in eight fresh sequential processes:
production at 256 MiB, inclusive 512 MiB, inclusive 2,032 MiB, ORT, ORT,
inclusive 2,032 MiB, inclusive 512 MiB, production at 256 MiB. The symmetry
balances a linear time trend; it does not prove the
absence of nonlinear drift or unrelated workstation load. All samples remain.

Each process uses the original twenty-clip corpus (213.265 seconds audio), one
complete warmup pass and three measured passes. There are 640 complete requests:
160 warmup and 480 measured. Every result, input and held-output check must pass.
Timers include frontend, graphs, greedy decoder and owned result construction;
loading, file IO and external validation are outside. The original managed and
native consumers remain byte-identical. ORT 1.29 CPU has one intra/inter thread,
sequential execution, all graph optimizations and spinning disabled. Managed
runtime settings remain normal. Affinity is CPU2 before process startup.

Each worker needs 14 GiB available RAM before launch and must remain below 12 GiB
RSS, 1,800 seconds, 1 GiB output, with at least 1 GiB available RAM and 20 GiB free disk.
The shared process monitor waits at most 900 seconds before refusing preflight.
Unrelated workstation processes are preserved. A missing/failed worker is not
replaced automatically, and no sample or warmup count changes after launch.

For each role, identical-process control limits are corpus latency max/min at
most 1.10 and every clip's per-process mean max/min at most 1.20. These descriptive
screens are fixed before inference. If any fail, candidate attribution is
rejected and no unchanged local timing repeat follows. Report all observed
means and variation with that verdict, without calling them calibrated speedups.

If all controls pass, a candidate qualifies for AMD follow-up only when its
mean complete-corpus latency is at least 5% below the production control and no
clip's aggregate latency is over 5% worse. Compare both candidates against the
same fresh production and ORT observations. The ratio to ORT is descriptive;
two processes per role cannot prove a 1.05 parity target or broad hardware behavior.
Prefer the lower memory budget if both meet these criteria and its corpus mean
is within 2% of the larger budget. Otherwise retain the faster qualifying budget.
If neither qualifies, retain both failures and choose a distinct mechanism.

The original three Windows native tensor failures remain open and separately
recorded, even when every public transcription agrees. No historical latency
is substituted into this comparison.
