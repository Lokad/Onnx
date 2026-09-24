# Observe Parakeet masking and padding across the complete corpus

M66 release qualification must finish before `prepare`, staging or launch.
This protocol requires an actual AMD build review and complete capture; local
source inspection and checker tests alone do not qualify the observer.

The [matched ORT analysis](../managed-phase-results/masking-padding-20260924.md)
attributes 6.063 seconds of excess time to masking and padding. This observer
establishes concrete layouts and mask values before choosing an optimization.
It records every one of the 72 float Where and 48 Pad operations during each
of the original 80 requests: 9,600 observations across all 20 clips and 24 layers.
Mixed masks and non-dense layouts are valid diagnostic findings.

The [source work counts](../managed-phase-results/masking-work-20260924.md)
quantify the predicted index calculations and logical writes for this corpus.
`work_counts.py` consumes retained evidence only; its closed counts neither
replace the capture nor establish instruction counts or a performance gain.

Core remains byte-identical to the measured M66 composition (`37c24375`). An
isolated Data build adds a disposable logger scope to the private graph-call
helper. Every node ordinal is checked; target inputs and subsequent outputs
are described without retaining activations. The logger is restored on exit.
The original consumer retains all requests and assertions; metadata is saved
after its request clock. Capture timings are diagnostic and cannot admit a change.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` followed by:

    tests/parakeet/masking-padding-layout-amd/run.py inspect
    tests/parakeet/masking-padding-layout-amd/run.py prepare
    tests/parakeet/masking-padding-layout-amd/run.py stage
    tests/parakeet/masking-padding-layout-amd/run.py launch build
    tests/parakeet/masking-padding-layout-amd/run.py observe build
    tests/parakeet/masking-padding-layout-amd/run.py collect build
    tests/parakeet/masking-padding-layout-amd/review_build.py
    tests/parakeet/masking-padding-layout-amd/run.py launch capture
    tests/parakeet/masking-padding-layout-amd/run.py observe capture
    tests/parakeet/masking-padding-layout-amd/run.py collect capture
    tests/parakeet/masking-padding-layout-amd/audit.py
    tests/parakeet/managed-phase-results/publish_masking_layouts.py

Observe the recorded owner until terminal before collecting. Build review must
prove the original Data helper body, consumer checks, all other original methods,
public surface and flags before capture. Build only on AMD, using the existing
SDK 10.0.204/runtime 10.0.8 and offline feed. CPU2 runs work; CPU0 monitors it.
Build limits are 3 GiB RSS/180 seconds per command, with 2 GiB available RAM and
1 GiB tmpfs before launch. Capture requires 11 GiB RAM/2 GiB tmpfs, is capped at
12 GiB RSS/900 seconds, and retains 1 GiB free RAM/tmpfs throughout. Total stage
output is capped at 512 MiB. No new model downloads or system changes are needed.

Local namespace: `artifacts/parakeet-masking-padding-layout-amd-20260924`.
Remote namespace: `/dev/shm/lokad-parakeet-masking-padding-layout-20260924`.
Existing output is refused; preserve failures. Repairing an analyzer must not
repeat inference. Write auditor stdout outside the artifact directory.
The detailed living plan is `.agent/m67-parakeet-masking-padding-diagnosis-20260924.md`.

The checker tests use retained AMD compiled instructions and deliberately
damage result checks, branches, exception regions and graph execution:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/parakeet/masking-padding-layout-amd -p test_checks.py -v

These validate the auditor; they do not qualify the unbuilt observer itself.

The publisher requires the actual diagnostic closure and verified raw collection.
It retains all 9,600 observations in CSV, groups every mask/layout finding, and
does not use capture clocks to admit an optimization.
