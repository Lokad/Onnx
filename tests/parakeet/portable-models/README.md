# Complete-model qualification of the portable arithmetic composition

Run from the repository root with the existing pinned model/reference assets:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/portable-models/prepare.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/portable-models/run.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/portable-models/audit.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/portable-models/shared.py prepare
    C:/Python313/python.exe -X utf8 -B tests/parakeet/portable-models/shared.py run
    C:/Python313/python.exe -X utf8 -B tests/parakeet/portable-models/shared.py audit

Each dependent command requires actual exit zero and terminal workers. These
commands have already completed successfully; do not restart closed runs.
The [result](results-20260922.md) records their identities and scope.

Products come from the closed `parakeet-portable-composition-20260922`
artifact. The tools verify its closure and separate external-operand proof.
Audio output goes to `artifacts/parakeet-portable-models-20260922`, shared
output to `artifacts/parakeet-portable-shared-20260922`. Preparations refuse
existing destinations and freeze all tool, consumer, model and input identities.

Native/public output errors must remain at most `1e-4` under
`abs(actual-reference)/max(1,abs(reference))`. Integer decisions, shapes,
input preservation, held outputs and repeat results retain their original
checks. No test hash is accepted before every corresponding native output
passes. This is correctness qualification, with no speed-selection gate.

CPU2 affinity is inherited before CLR startup, monitor CPU0. Builds require
8 GiB available memory; native/shared/Pyannote inference requires 10 GiB,
with an 8 GiB aggregate RSS bound. The public Parakeet lane requires 14 GiB
available and has a 12 GiB RSS bound. Available memory must stay above 1 GiB,
disk above 20 GiB, output below 1 GiB. Workers have 900-second bounds except
Parakeet native/public at 1,200 seconds. The original monitor preserves
preflight waits, worker births, descendants, samples and terminal state.

The following normal source and package qualification is separate:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/portable-suites/run.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/portable-suites/audit.py

It archives root `1b36c548`, applies the two exact arithmetic source files and
adds only the completely native-qualified DINOv3 hash pair to its test copy.
Normal project references must preserve all 3,109 Core and 697 Data methods
and public declarations against the model-qualified runtime. Full backend
and tensor suites and the independent NuGet consumer must pass. Production
source and historical fixtures remain unchanged by these commands.
