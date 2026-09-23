# Full qualification of the isolated Winograd product

M33 component screen 302de6c9 and normal build 50e50f01 admit this campaign.
Source v2 3c7e4d24 retains every root file, adds the exact qualified arithmetic,
optional shared-budget weights and stride-one dispatch with direct fallback.
The first build's scope failure 1e3e9d40 is retained; v2 confines changes to
seven existing methods and sixteen additions, with public/Data declarations exact.

Sixteen fixed jobs: SDK, CLI restore/build, backend restore/build, tensor
restore/build, complete candidate method inventory, all backend tests, all
tensor tests, focused graph tests in AVX2 and AVX512, package, independent
consumer restore/build/run. The normal-build inventory must match all 3,179
Core and 697 Data methods. All 3,432 original backend cases and 343 tensor
cases remain, with the same 41 existing AMD skips; 17 new cases are required.
The two focused workers must each execute all 48 convolution graph cases.
Only the AVX2 worker disables AVX512; no other optimization override is used.

The PackageReference consumer preserves the original direct-only graph with
18,432 prepared bytes and 8,384 scratch bytes. A separate eligible graph
requires 51,200 prepared bytes and 28,800 scratch bytes, exact dyadic-oracle
values and independently owned repeat/held outputs. This proves actual
packaged optional preparation and dispatch, not just isolated private calls.

Worker CPU2, supervisor CPU0, SDK10.0.204/runtime10.0.8, preflight10GiB available
and3GiB tmpfs, live8GiB RSS ceiling,1GiB available/tmpfs floors,900seconds/job,
1GiB output and2GiB artifact ceilings. Freeze all sources and dependencies.
Use run.py prepare,stage,launch,observe,terminal-only collect,then audit.py.
No application performance or Microsoft ORT ratio follows from this campaign.
Full model/native/public/meeting and matched application checks still apply.
