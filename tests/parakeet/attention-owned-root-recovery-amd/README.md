# Finish attention preparation release qualification after a test-source repair

The original root campaign failed at the normal tensor suite. Its new portable
fixture declared five optional parameters on its `Graph` helper, violating the
unchanged repository source policy. Backend passed 3,597 cases with 43 skips;
tensors passed 393 with that one failure. Later jobs did not run. Original
collection and `failed.json` remain immutable; there is no successful closure.

This recovery makes every helper argument explicit using its original default.
It changes no product source, test assertion, case name, source-policy rule,
runtime option, numerical kernel or performance input. The measured candidate
and all completed performance campaigns remain unchanged. The original focused
fixture is retained byte-for-byte; `fixture_repair.py` checks the exact conversion.
All 445 other source inputs must equal the measured snapshot, including product
and source-policy files. Failure receipt `9c757ee0` binds the preserved failed run.

Use the original 18 jobs and complete root/package auditor. Expected full suites
are backend 3,597 passed / 43 skipped and tensors 394 / 0 normally; backend
3,507 / 133 and tensors 394 / 0 with AVX512 disabled. All 3,288 Core and 697 Data
method bodies, flags and public metadata must equal the measured product. NuGet
contents and independent package consumption must pass. Do not rerun any closed
performance campaign for this test-only correction.

From the repository root, prefix commands with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/attention-owned-root-recovery-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/attention-owned-root-recovery-amd -p test_*.py
    tests/parakeet/attention-owned-root-recovery-amd/apply_integration.py
    tests/parakeet/attention-owned-root-recovery-amd/source_scope.py
    tests/parakeet/attention-owned-root-recovery-amd/run.py prepare
    tests/parakeet/attention-owned-root-recovery-amd/run.py stage
    tests/parakeet/attention-owned-root-recovery-amd/run.py launch
    tests/parakeet/attention-owned-root-recovery-amd/run.py observe

Freeze tools before prepare. Apply, prepare, stage and launch once. Observe the
same owner to terminal, then collect once and execute audit.py once, directing
audit console output outside the artifact. Preserve any failure. Prior source
application and failed qualification must never be replayed.

Artifact: `artifacts/parakeet-attention-owned-root-recovery-amd-20260928`.
Integration: `artifacts/parakeet-attention-owned-root-recovery-integration-20260928`.
VM: `/dev/shm/lokad-attention-owned-root-recovery-20260928`.
Retain CPU2 compute / CPU0 monitoring; 10 GiB available RAM / 3 GiB tmpfs
preflight, 8 GiB owned RSS, 1 GiB free, 900 seconds/job and four hours overall.
Use the VM's SDK 10.0.204 and offline feed with `--tl:off`; no Windows .NET build.
The last local inventory is 49.656 decimal GB uniquely allocated.

Only successful actual-root/package qualification permits source and benchmark
promotion. The candidate's admitted Parakeet result remains 44.97562548 seconds
versus ORT 39.2002844835: 1.147329058 times ORT and 1.128490% matched improvement.
The <=1.05 objective is still open. Fresh request attribution must guide the
next causal hypothesis; no alternative optimization is part of this recovery.
