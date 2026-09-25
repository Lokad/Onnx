# Qualify the corrected root without weakening repository design checks

The original root run is failed at f129f7d8. All numerical backend cases and the
compiled product comparison passed, but tensor source checks rejected the extra
Data friend assembly and optional parameters in two test helpers. Preserve the
entire failure, including 392 passed and two failed tensor cases. Its last six
jobs did not run. Do not resume or relabel that campaign as successful.

The four-file correction removes Data friendship, exposes and documents the
existing PrepareOwnedMatMulWeights opt-in method, and expands all 49 test-helper
calls to the exact argument values previously supplied by the compiler. Neither
source-policy guard, test assertion, test name nor numerical implementation changes.
All 435 original inputs remain; source_scope.py derives the correction from the
frozen source and verifies the actual current bytes. No old evidence is modified.

Keep the original sixteen root/package jobs and add only restore/build of the
existing complete inventory helper. Its unchanged source emits both public
surfaces and assembly attributes. Require all 3,281 Core and 697 Data method
bodies, resolved operands, locals, exception regions and implementation flags
to match the measured candidate. Only one Core method becomes public, and only
the Data friend attribute is removed. Data declarations and attributes stay exact.
The strict metadata tests reject other public or assembly changes.

Require the original complete suite census: backend 3,546 pass / 42 skip normally
and 3,456 pass / 132 skip with AVX512 disabled; tensors 394 pass / zero skip in
both modes. Both source-policy tests must pass unchanged. Retain every original
NuGet dependency, import, matrix, convolution, ownership and PackageReference check.
Preserve all original warnings and require no warnings from the inventory helper.

Use SDK 10.0.204, --tl:off, and the canonical offline feed on the exclusive AMD VM.
Keep 10 GiB RAM / 3 GiB tmpfs preflight, 8 GiB RSS, 1 GiB remaining, 900 seconds
per job and four hours total. CPU 2 computes; CPU 0 monitors. No Windows build.

Use C:/Python313/python.exe -X utf8 -B with source_scope.py, consumer_scope.py and
unittest discovery first. Run bind_integration.py once, then run.py prepare,
stage, launch, observe while live, collect once terminal and audit.py once.
Keep audit stdout outside the artifact directory. Tools freeze at preparation.
No new performance score follows this source-policy qualification.

Local: artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925.
Source correction: artifacts/parakeet-owned-batch-isolation-root-policy-20260925.
Remote: /dev/shm/lokad-parakeet-owned-batch-isolation-root-policy-20260925.
