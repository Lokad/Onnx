# Complete models for the new Parakeet / single-panel composition

Candidate Core `abbf5e98` / Data `eb452663` comes from closed operator/source
composition `17e10dce`. Production remains unchanged. The exact new candidate
must pass every original numerical, public, input/repeat and held-output check.
No tolerance, corpus, runtime setting, packing cap or speed gate changes.

From repository root, run sequentially with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/single-panel-models/prepare.py
    tests/parakeet/single-panel-models/run.py
    tests/parakeet/single-panel-models/audit.py
    tests/parakeet/single-panel-models/shared.py prepare
    tests/parakeet/single-panel-models/shared.py run
    tests/parakeet/single-panel-models/shared.py audit

Each dependent command requires actual exit zero and terminal owners. Fresh
artifacts are `parakeet-single-panel-models-20260922` and
`parakeet-single-panel-shared-20260922` under artifacts; existing paths are refused.

Coverage: 784 Parakeet native arrays / 3,090,494 values, twenty public clips,
18 Pyannote arrays / 2,917,107 values and sixteen diarization requests, then
166 shared-model arrays / 5,000,814 values. Native scaled error stays <=1e-4.
Pyannote arrays and public results also must match the selected single-panel
candidate exactly; its frozen closure and result are pinned before execution.
Only the GraphQualification consumer's expected Data literal changes, with
compiled inspection requiring its other 95 methods and all declarations unchanged.

CPU2 before CLR, monitor CPU0. Build preflight/RSS 8 GiB, model preflight 10 GiB
and RSS 8 GiB; public Parakeet preflight 14 GiB and RSS 12 GiB. Keep 1 GiB available,
20 GiB disk and 1 GiB output guards. Audio workers are bounded to 1,200 seconds;
Pyannote/shared/build workers to 900 seconds. All .NET commands disable terminal
logging. Shared regression retains its existing fingerprint-disabled diagnostic
setting; timing will use normal runtime settings. Full suites/package, long
meetings and fresh AMD performance qualification remain separate requirements.
