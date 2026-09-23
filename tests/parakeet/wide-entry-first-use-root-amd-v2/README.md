# M54 normal root qualification, corrected hardware census

The exact measured source is already integrated at `81f75c38`: 422 inputs,
with one modified file and two additions. Do not integrate again. Preparation
verifies every source byte and all admitted Parakeet, Pyannote and graph gates.

V1 stopped in its checker after eleven successful processes because its
AVX512-disabled census omitted ten existing Exp512 tests. The source requires
Vector512 hardware for those four facts and six theory cases. Its complete
[incident record](../wide-entry-first-use-results/root-census-correction-20260923.md)
is preserved and required by V2 preparation. V1 is not a release qualification.

V2 retains all sixteen jobs, worker commands, SDK/runtime, method inspection,
ordinary suites, package consumer and resource limits. Only the five explicitly
named Exp512 methods and the corresponding disabled counts change, alongside
namespace/dependency labels. Ordinary backend remains 3,449 passes/41 skips;
AVX512-disabled backend is 3,359 passes/131 skips. Exactly 93 hardware cases
become skipped and three unsupported-hardware cases become active. Every other
per-test outcome must remain exact. Both tensor modes require 343 passes/0 skips.
No product or test source is changed by this correction.

A normal SDK 10.0.204 build must preserve every one of 3,189 Core and 697 Data
method bodies, implementation flags and public declarations against the measured
candidate. The real NuGet PackageReference consumer must pass import, arithmetic,
prepared spatial/Winograd execution, unchanged inputs and owned-result checks.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py`. Each output namespace must be new. Local
artifacts use `parakeet-wide-entry-first-use-root-amd-v2-20260923`; the VM uses
`/dev/shm/lokad-parakeet-wide-entry-first-use-root-v2-20260923`.
Compute CPU 2, monitor CPU 0; 10 GiB available/3 GiB tmpfs preflight, 8 GiB RSS,
900 seconds/job and four hours/campaign. Build and infer only on AMD, with
`--tl:off --nologo -v minimal`. No scored timing is repeated.

Run `consumer_scope.py` for the exact retained-worker/checker delta. Sixteen
local checker tests, including actual TRX and adversarial outcome changes, run
with `python -B -m unittest discover -s tests/parakeet/wide-entry-first-use-root-amd-v2
-p test_*.py -v`. They qualify the checker; actual full V2 execution remains required.
