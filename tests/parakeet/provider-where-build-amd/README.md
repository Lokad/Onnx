# Provider-only Where: isolated AMD build

Freeze and build source64460d04 from selected release81f75c38. This is a new
candidate after rejected generic insertionfd847767 and untimed code review
1cf6a124/4f8b3564. No product root changes and no timing occurs here.

Four serial CPU2 jobs:SDK10.0.204,offline CLI restore,normal Release CLI build,
and retained Bridge method inventory against measured Core672e5f30/Data065b7a7f.
The supervisor remains onCPU0. Every dotnet build/restore uses --tl:off --nologo
-v minimal. Runtime10.0.8, original offline feed, no Windows build or inference.

Only CPUExecutionProvider.Where may change among existing methods. Its Float arm
changes from15instructions to48, adding four typed locals; all196original
instructions outside that arm and every branch/switch target retain their exact
meaning. Require the4096-element/scalar guards, exact casts and call arguments,
one profiling-stage entry, Try success with owned OpResult, and original tensor
fallback. The full provider body must match the prospectively reconstructed
229instruction body, including local signatures, maxstack and exception census.
Every3188otherCore method, including generic Tensor.Where, all697Data methods,
existing flags and public surfaces must remain exact. Compiler renaming is refused.
The only new method must be the exact193instruction V3 uniform helper with flags520.
The reviewer pins its entire proven body, not just a few expected calls.

The transport/supervisor/resource/inventory decoder are the frozen M56 build
tools with namespace substitutions only; their old M56 comments do not define
this candidate's scope. Source preparation checks that equivalence. New scope
checks and mutation tests define the provider-only change. Pin the selected root,
M56 failure and diagnostic, source receipt and helper proof before staging.

Run C:/Python313/python.exe -X utf8 -B with:

    tests/parakeet/provider-where-build-amd/test_checks.py
    tests/parakeet/provider-where-build-amd/run.py prepare
    tests/parakeet/provider-where-build-amd/run.py stage
    tests/parakeet/provider-where-build-amd/run.py launch
    tests/parakeet/provider-where-build-amd/run.py observe
    tests/parakeet/provider-where-build-amd/run.py collect
    tests/parakeet/provider-where-build-amd/audit.py

Freeze before prepare; refuse existing namespaces and preserve failed attempts.
Observe only a live unclosed lane. Collect after every PID/birth owner is terminal.
Namespaces:artifacts/parakeet-provider-where-build-amd-20260924 locally,
/dev/shm/lokad-parakeet-provider-where-build-20260924 onAMD. Preflight12GiBavailable/
3GiBtmpfs;8GiBRSS,1GiBremainingmemory/tmpfs,900s/job,fourhours/campaign,
1GiBoutput/job,2GiBcampaign. Do not lower these limits. Review closed cache or
duplicate-file retention before staging if headroom is insufficient.

Admission here permits numerical qualification through both tensor and provider
boundaries, with provider error/option and4095/4096/4097boundary cases. It does
not admit component timing, application timing, root integration or benchmark changes.
