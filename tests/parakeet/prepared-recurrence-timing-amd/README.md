# Complete captured Parakeet LSTM timing

This prospective component comparison follows successful compiled-scope review,
300 focused contracts, actual decoder/call qualification `20b3bc4b`, and complete
native/public trajectory qualification `c0b2f614`. Products are selected Core672e5f30/Data065b7a7f and
candidate Core3c23b44a/Datacc37b19e; only the common timing consumer is built.

The sample bank is all 380 actual calls from capture28c7afe4, in its original
case/step/node order: English37, French29, JFK46, token-limit4, frame-limit37 and
English-repeat37 steps, with both LSTMs for every step. W/R/B remain initializers
and only the original Y/Yh/Yc are outputs. Each graph retains a 64 MiB budget;
candidate prepares 13,107,200 bytes per graph and selected zero. Float declarations
use the exact captured shapes, including the input metadata absent in the first
correctness helper.

Run selected0, candidate0, candidate1, selected1 in normal mode, then the same
order with AVX512 disabled. Every fresh CPU2 process uses five warmup passes and
five measured passes over the entire fixed bank. Keep all 30,400 individual
clocks, including warmups, and their per-thread allocation counts. No trimming,
adaptive warmup, selective cases or unchanged retries. All original output bits
must equal capture; all inputs/constants remain unchanged and held outputs must
survive later calls and resets. Exact bits preserve the already-qualified native
1e-4 numerical bound.

Each call clock includes context reset, normal graph validation, allocation and
complete LSTM execution. File IO, fixture decoding, caller feed creation, output
hashing and report serialization are outside these clocks. Separately retain
each graph's cold construction, preparation and execution-context setup clocks
and allocations. Report both cold setup observations per product/mode; do not
infer precise cold-start latency from only two samples. Prepared arrays remain
resident across calls; no weight preparation may be hidden inside untimed warmup.

The seven fixed timing groups are the full 380-call corpus and each of the six
nonempty captured cases. Within each process, require measured-pass max/min
<=1.10 for the corpus and <=1.20 for every case. Between duplicate processes of
each product/mode, apply the same limits to the exact five-pass means. These are
42 repeatability controls per mode, 84 total. Require both modes to pass every
control, at least 10% lower candidate corpus time, and no case over 5% slower.
Means and ratios use raw ticks and exact rational arithmetic. A valid slower
result is a preserved rejection. A failed control invalidates the comparison;
neither permits an unchanged retry.

Record foreign-process CPU accounting using the existing application monitor and
auditor, and require observed foreign CPU fraction <=0.01. Paired snapshots can
miss short-lived processes; retain that limitation and all repeatability controls.
Require exact PID/birth ownership, every worker/thread on CPU2 and monitoring
on CPU0. Limits are 8 GiB available/2 GiB tmpfs before each job, owned RSS<4 GiB,
minimum available/tmpfs1 GiB, output<=128 MiB/job, artifacts<=1 GiB/campaign,
elapsed<900s/job and four hours/campaign. Use SDK10.0.204, runtime10.0.8, the
existing offline feed and canonical global.json. No Windows inference or build.

A passing component result permits the unchanged six-process complete application
comparison (current,candidate,ORT,ORT,candidate,current), requiring all 63 controls,
at least 3% corpus gain and no clip over 5% slower. It does not promote the product
or establish Parakeet parity. The selected BENCHMARK.md remains unchanged until
the application and release regression gates pass.

From the repository root, run these with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/prepared-recurrence-timing-amd/test_checks.py
    tests/parakeet/prepared-recurrence-timing-amd/run.py prepare
    tests/parakeet/prepared-recurrence-timing-amd/run.py stage
    tests/parakeet/prepared-recurrence-timing-amd/run.py launch
    tests/parakeet/prepared-recurrence-timing-amd/run.py observe
    tests/parakeet/prepared-recurrence-timing-amd/run.py collect
    tests/parakeet/prepared-recurrence-timing-amd/audit.py

Outputs are exclusive-create under local
`artifacts/parakeet-prepared-recurrence-timing-amd-20260924` and VM
`/dev/shm/lokad-parakeet-prepared-recurrence-timing-20260924`. Collect only terminal
PID/birth owners; never observe a closed namespace. Preserve any failure before
diagnosing it. Only the consumer is built, against already-qualified products;
reuse immutable fixtures through hardlinks and unlink before replacements.
