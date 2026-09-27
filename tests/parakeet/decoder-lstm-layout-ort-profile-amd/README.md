# Refresh ORT attribution for the exact Parakeet application

This adapter binds successful root/package closure efb99eea of source edc1a6c8,
after graph and Pyannote application admission. No stage, inference, new kernel
or observer build has occurred through this adapter at this preparation point.

Reuse tests/parakeet/ort-diagnosis-amd/run.py, observer.py, remote.py and analyze.py
without changing their functions. Only application and output locations change.
The inherited prospective-plan file describes that original observer; this
adapter's qualification.json retains the new root, application and adapter pins.
The application closure is e3ba182a: its matched ORT corpus latency is 39.513180
seconds and the candidate is 46.931723 seconds. The actual built product must
match all 3,286 measured Core methods and 697 Data methods before staging.

Run the original control/profile pair: eighty public requests per process,
twenty warmups and sixty measurements, on the same twenty clips. Preserve
original graph outputs, decoding, numerical checks, runtime, thread counts and
graph optimizations. Each process covers 4,960 graph calls. Reconcile every
request and node event, including zero-duration events, nested intervals and
work outside nodes. Keep control-versus-unprofiled and profile-versus-control
ratios separate. Profiler clocks are attribution, not a new benchmark result.

The unchanged worker requires 12 GiB available RAM and 3 GiB free tmpfs before
each process, 12 GiB maximum RSS, at least 1 GiB RAM/tmpfs remaining, 1 GiB maximum
output and 900 seconds per process. Verify sufficient headroom before staging;
do not lower a limit if it fails. Compute stays on CPU 2 and monitoring on CPU 0.
Only one VM workload may run. Collect after all owners are terminal, audit once
with stdout/stderr outside the artifact, and preserve every failure and clock.

From repository root, prefix commands with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/decoder-lstm-layout-ort-profile-amd/run.py review
    tests/parakeet/decoder-lstm-layout-ort-profile-amd/run.py stage
    tests/parakeet/decoder-lstm-layout-ort-profile-amd/run.py launch
    tests/parakeet/decoder-lstm-layout-ort-profile-amd/run.py observe
    tests/parakeet/decoder-lstm-layout-ort-profile-amd/run.py collect
    tests/parakeet/decoder-lstm-layout-ort-profile-amd/analyze.py

Review requires the exact successful ROOT_DIGEST. Before staging, verify that the
original instrumentation hook accepts the retained native consumer and freeze
the adapter. Bind the new full managed profile alongside this native capture;
do not use the old managed partition to rank current excesses. Inspect exact
ORT dispatch and data movement for the resulting leading excess before choosing
one falsifiable implementation change. Kernel sampling is a separate decision
only if ordinary node attribution and pinned source leave a relevant ambiguity.

Local: artifacts/parakeet-decoder-lstm-layout-ort-profile-amd-20260927.
VM: /dev/shm/lokad-lstmlayout-ort-profile-20260927.
