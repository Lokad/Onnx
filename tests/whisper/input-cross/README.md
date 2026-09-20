# Whisper encoder crossed inputs

Compare both encoders on each of the two saved feature arrays from the fixed
twenty-recording numerical corpus, plus its first-recording repeat. This isolates
input sensitivity from engine differences while holding the original model and
managed core fixed. It does not qualify a changed public pipeline or measure speed.

`prepare.py --artifact <directory>` verifies the closed reference receipts and pins
every reused array, model, tool, binary and native ORT library. First build
`Probe.csproj` in Release with `--tl:off --nologo -v minimal`,
`-p:FrozenProductDirectory=<absolute artifacts/asr-labeled-20260919/managed-bin>`
and `-o <artifact>/bin`. Use the existing .NET 10.0.12 runtime.

Run `test_audit.py`, then `run.py --artifact <directory> --engine managed` and,
only after the original supervisor and children exit, the same command with
`--engine native`. Every one of 21 baseline outputs per engine must reproduce its
saved bytes before its corresponding crossed-input call. Outputs and logs are
single-use. The native worker uses the original reference Python environment.
The supervisor enforces CPU affinity and explicit memory/time guards.

Run `audit.py --artifact <directory> --output <artifact>/audit.json`, followed by
`close.py --artifact <directory>`. The audit independently reads all full arrays,
checks original/held/repeated outputs and input identities, reconstructs two exact
algebraic decompositions, and checks resource samples and actual process births.
Exit zero means a completed diagnostic; numerical acceptance is reported separately.

All six difference summaries use the original native-output denominator. Neither
engine is an independent higher-precision reference. Preserve numerical failures
and do not interpret component magnitudes as causal percentages.
