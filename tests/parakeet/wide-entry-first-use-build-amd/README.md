# Wide matrix entry compiled with full optimization on first use

M54 builds a separate wide matrix entry and optimizes it and the existing
private packing helper on first use. Source-v2 receipt72c22bee fixes a missing
static import found during review of unexecuted source-v1. Root product remains
unchanged. No numerical or performance admission follows this build.

The original entry body/flags remain exact. Four caller operands target a new
guarded dispatcher. The clone, its parallel lambda and closure constructor must
match the originals after only the declared closure/name substitution. The
private helper changes flag8to520 only. All other existing bodies/flags and all
697Data methods/public APIs remain exact. Expect3185old/3189new Core methods.
The dispatcher source is frozen and reviewed for null/rank checks, all wide
guards and exact fallback; the compiled gate additionally checks its targets
and threshold constants. It does not claim independent semantic verification
of every new dispatcher instruction; numerical boundary tests remain required.

Run `C:/Python313/python.exe -X utf8 -B -m unittest discover -s
tests/parakeet/wide-entry-first-use-build-amd -p test_checks.py`, then run.py
prepare, stage, launch, observe, collect separately and audit.py after all
owners terminate. Commit frozen tools before execution. Four serial jobs:
SDK identity, offline restore, normal Release build and full method inventory.

CPU2 compute/CPU0 monitoring. Bounds:10GiBavailable/3GiBtmpfs preflight,
8GiBRSS,1GiBminimum available/tmpfs,1GiBoutput/job,2GiBcampaign,900seconds/job,
four hours overall and monitoring gaps<10seconds. Existing namespaces refused.
