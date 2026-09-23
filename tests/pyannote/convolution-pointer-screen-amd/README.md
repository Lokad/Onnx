# M28 prospective complete-call screen

Compare selected Core208371f6 with numerically/codegen-qualified Coree776cec2.
Reuse unchanged SpatialWeightScreen and qualified LayerGraphs callers, invoking
reflection helpers rather than candidate-specific Main. No rebuild, profiling,
ISA/tiering overrides, calibration, trimming or unchanged timing retry.

Four fresh AMD CPU2 processes run current/candidate/candidate/current. The same
108graphs and geometry-derived1074iterations/pass, one warmup and three measured
passes retain17184clocks (4296warmup/12888measured) and512preparation clocks.
Complete-call timers include assertions/dispatch recording, finite scans,
scratch, conversions, kernel and epilogues; exclude hashing/journal IO.

Prospective rules were defined before any M28 timing: all32repeatability
controls <=1.10aggregate/<=1.20eachform; candidate/current<=.98aggregate and
<=1.05each eligible form; additionally max(candidate process aggregate) must
be strictly less than min(current process aggregate). Every rule is mandatory.
Exact integer-clock fractions decide boundaries. Fifteen scorer tests pass,
including one-tick limits, equal/overlapping processes and both role members.
Historical verdicts are not rescored. Application qualification still requires
at least3%complete-dialogue improvement and all original crop/native/ownership/
repeatability/package/meeting/recovery gates before integration.

CPU0monitor; original12GiBavailable/3GiBtmpfs preflight,8GiBRSS,
1GiBliveavailable/tmpfs/output,900seconds/job,2GiBartifacts. No overlapping
workload. Python3.13 -X utf8 -B run.py prepare,stage,launch,observe,collect,
then audit.py. Artifact artifacts/pyannote-convolution-pointer-screen-amd-20260923;
VM /dev/shm/lokad-pyannote-convolution-pointer-screen-20260923. Refuse existing
destinations, preserve all failures and clocks, collect terminal owners only,
never observe after closure. This screen is not application/ORT parity.
