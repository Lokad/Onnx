# Qualify the measured padding change as the actual release package

This adapter is prepared locally while graph qualification runs. Do not apply,
prepare or stage it until the fresh graph comparison and Pyannote application
comparison both close with admission. It selects no new optimization.

The source scope is the 435 qualified root inputs, the measured Pad dispatch
change and private helper, and six Pad facts. The integration fixture removes
one optional helper default and supplies the same three false arguments
explicitly. All assertions, test names and source-policy tests remain unchanged.
There are 437 resulting build inputs and three changed paths. The root is
checked before and after applying them, including unexpected compilation inputs.

The 18 existing root jobs, worker, transport and independent auditor are reused.
Compile once on the AMD VM with SDK 10.0.204, the existing offline package feed
and --tl:off. Compare every one of the 3,282 Core and 697 Data compiled methods
to the measured a74acb17 / be954dc4 candidate: bodies, operands, locals,
exception regions, implementation flags, public surface and assembly attributes
must all match. No metadata exception remains from the previous root campaign.

Use the actual qualified root TRX for each hardware mode. Require exactly six
additional passing Pad facts, preserving every original name and outcome:
backend 3,552 passed / 42 skipped normally; 3,462 / 132 with AVX512 disabled;
tensors 394 / 0 in each mode. Keep the exact warning census, unchanged tensor
source archive, package DLL/dependency checks and independent PackageReference
consumer's numerical, graph-preparation, import and held-output checks.

The prerequisites bind both actual DLLs through complete Parakeet/Pyannote/shared
correctness, 63 Parakeet repeatability controls and 21 performance gates, the
fresh eight-case graph admission, and all four Pyannote timing cases plus the
two 600-second meetings and recovery request. Closed failures remain intact.
Unit-test fixtures containing retained timings are schema tests only; they are
never written as campaign evidence or used to authorize integration.

From repository root, prefix these commands with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/pad-current-root-amd/source_scope.py
    tests/parakeet/pad-current-root-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/pad-current-root-amd -p test_*.py

Only after every prerequisite is admitted:

    tests/parakeet/pad-current-root-amd/apply_integration.py
    tests/parakeet/pad-current-root-amd/run.py prepare
    tests/parakeet/pad-current-root-amd/run.py stage
    tests/parakeet/pad-current-root-amd/run.py launch
    tests/parakeet/pad-current-root-amd/run.py observe
    tests/parakeet/pad-current-root-amd/run.py collect
    tests/parakeet/pad-current-root-amd/audit.py

Freeze tools at preparation. Only observe repeats while an owner is live.
Collect/audit once after all owners are terminal; save audit stdout outside the
artifact folder. Preserve failures and partial integration/collection receipts;
do not restart a failed campaign unchanged. The original Shape.cs is backed up
under the integration artifact before writing. A partial integration is reviewed
against that backup and the frozen expected source map before any next action.

Keep CPU 2 compute, CPU 0 monitoring, 10 GiB RAM / 3 GiB tmpfs preflight,
8 GiB owned RSS, 1 GiB remaining RAM/tmpfs, 900 seconds per job and four hours
overall. No Windows .NET execution, model downloads or new performance score.

Local campaign: artifacts/parakeet-pad-current-root-amd-20260926.
Integration: artifacts/parakeet-pad-current-root-integration-20260926.
VM: /dev/shm/lokad-parakeet-pad-current-root-20260926.
