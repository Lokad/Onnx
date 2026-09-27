# Refresh managed attribution after pointwise release qualification

This adapter shares the native adapter's actual root admission `fc116763`.
Source `c28b3848` is qualified and committed; built Core is `a6f7d9f9`, Data
`1ba343fd`. Qualification is rechecked before preparation or inference.
Measured Core is `7cac6788`, Data
`dd56902f`; source receipt `9be6d038` contains the one remainder-sharing change.

Compatibility review `e234eec8` proves reuse of consumer `38ab5c7e` and observed
Data `51b6d2bb`: all 697 Data methods/flags and both public surfaces/assembly
attributes remain preserved. Actual-root qualification must additionally match
all 3,288 Core and 697 Data methods. Use that actual built Core in every mode,
its Data for control, and the retained Data observer for phase and wall.

Reuse packed-final-row-profile-amd's transport/worker and the complete
owned-batch-isolation-profile-amd auditor. No request consumer, decoding loop,
node timer or observer is rebuilt or changed. Keep control, phase and wall
processes, each with eighty requests: twenty warmups and sixty measurements
over the original twenty clips. Require exact admitted complete public results,
held-output ownership and complete reconciliation of phase/node intervals and
remaining work. Report phase/control and wall/phase separately; never subtract
overhead or use profiler clocks as BENCHMARK.md results.

Keep CPU2 compute / CPU0 monitoring, 11 GiB available RAM / 2 GiB free tmpfs
before each process, 12 GiB maximum RSS, 1 GiB remaining RAM/tmpfs, 512 MiB output
and 900 seconds/process. Only one VM workload may run. Preserve every failure.

After actual root admission is bound and the adapter is frozen, prefix with
C:/Python313/python.exe -X utf8 -B from repository root:

    tests/parakeet/pointwise-tail-profile-amd/run.py prepare
    tests/parakeet/pointwise-tail-profile-amd/run.py stage
    tests/parakeet/pointwise-tail-profile-amd/run.py launch
    tests/parakeet/pointwise-tail-profile-amd/run.py observe

Follow the same owner to terminal, then collect and audit.py once with console
output outside the artifact. Combine the fresh managed and native profiles,
retaining complete node membership. No further optimization is selected yet.

Local: artifacts/parakeet-pointwise-tail-profile-amd-20260927.
VM: /dev/shm/lokad-pwt-profile-20260927.
