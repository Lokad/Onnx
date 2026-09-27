# Qualify the measured pointwise remainder change as the release package

This adapter is prepared locally. Source application and VM work remain gated
on the actual admitted eight-case graph comparison and complete Pyannote
application comparison. Their closure digests are deliberately unset until
those runs finish and pass. No further optimization belongs in this integration.

The measured candidate is Core `7cac6788` / Data `dd56902f`; current is Core
`47984318` with the same Data. The independent complete Parakeet comparison
passes all 63 controls and 21 gates: 45.332653922 seconds versus current
46.320466422 and Microsoft ORT 39.232132721. The matched gain is 2.132562%; the
candidate/ORT ratio is 1.155498. The earlier component timing screen remains
failed and supplies no qualified speedup claim.

Bind the 443 qualified root inputs to source receipt `9be6d038`, which changes
only MathOps.cs and adds MathOps.PackedColumnTails.cs. Add the exact portable
PackedColumnRemainderTests.cs fixture `2b47d1c2`: 445 final source inputs and
three applied paths. Back up MathOps.cs before source application. Preserve all
other source bytes, including the existing source-policy test.

The two new portable facts cover 111 raw geometries, including widths 0–31,
row and column panel boundaries, reduction boundaries and observed widths
222/224/225. Small integer matrix sums provide an independent reference for
three accumulations. Check guarded offsets, unchanged inputs and packed data,
and zero repeated-call allocation. The fixture has not yet been compiled or
executed; both full hardware-mode suites below must actually run it.

Preserve original failed numerical closure `58cb4026`. The subsequent baseline
diagnosis demonstrates unstable NaN payloads even between identical builds.
Corrected arithmetic closure `558f2a52` passes 10,392 raw cases and 24 scalar
cases, requiring exact non-NaN bits and matching NaN classification. Retain all
payload differences and the failed history. Separate code-generation review
`3715408c` verifies eight accumulators, no vector spills and the original FMA
versus multiply/add behavior in both hardware modes. This does not relax model
agreement or complete public-result checks.

Reuse the original 18 root jobs, worker, transport and package auditor. Build
once on the AMD VM with SDK 10.0.204, the existing offline feed and --tl:off.
All 3,288 Core methods and 697 Data methods must match the measured candidate,
including bodies, operands, locals, branches, exceptions, flags, public surface
and assembly attributes. Consumers must load the actual built binary, and the
NuGet package must contain that same Core.

Require the complete previous census plus exactly the two passing facts:
backend 3,568 passed / 42 skipped normally and 3,478 / 132 with AVX512 disabled;
tensors remain 394 / 0 in both modes. Preserve every previous name and outcome,
the tensor-source archive, exact warning census, sole Google.Protobuf 3.33.5
dependency and independent PackageReference consumer's numerical, model import,
prepared-graph and held-output checks. Local synthetic fixtures test the Python
validators; they do not qualify future builds or performance results.

Old prerequisite VM namespaces retain their original payload and collection
metadata, measured products, models and consumers. Verify these and terminal
owners; use local proofs for retired historical output copies. For retained
trees on another filesystem, resolve the original path and copy with verified
bytes; otherwise preserve hard links. Verify every fresh prerequisite payload
and external input. Do not download or execute .NET on Windows.

From repository root, prefix each command with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/pointwise-tail-root-amd/source_scope.py
    tests/parakeet/pointwise-tail-root-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/pointwise-tail-root-amd -p test_*.py

After actual graph/application admission, bind both closure digests, freeze the
adapter and ensure all previous VM owners are terminal:

    tests/parakeet/pointwise-tail-root-amd/apply_integration.py
    tests/parakeet/pointwise-tail-root-amd/run.py prepare
    tests/parakeet/pointwise-tail-root-amd/run.py stage
    tests/parakeet/pointwise-tail-root-amd/run.py launch
    tests/parakeet/pointwise-tail-root-amd/run.py observe

Follow that same owner to terminal, then collect and audit once. Keep audit
console output outside the artifact. Preserve any failure. Keep CPU2 compute /
CPU0 monitoring, 10 GiB RAM / 3 GiB tmpfs preflight, 8 GiB owned RSS, 1 GiB
remaining RAM/tmpfs, 900 seconds/job and four hours overall. Check space for
complete outputs before staging. Update BENCHMARK.md only after the actual
root build and package pass.

Local: artifacts/parakeet-pointwise-tail-root-amd-20260927.
Integration: artifacts/parakeet-pointwise-tail-root-integration-20260927.
VM: /dev/shm/lokad-pwt-root-20260927.
