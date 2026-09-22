# Conditional M22 integration

`prepare_patch_v3.py` creates a reviewed five-file patch from the exact source
that passed ordinary Linux suites, method comparison and package consumption.
It checks every qualified source file against the root tree and permits only
the three LSTM source changes and two focused test changes. Known source text
normalizes line endings; binary files remain byte-exact. It verifies the three
existing files against selected source `9533cd67` and checks that the patch
applies. It changes no root product file and runs no build or inference.

Run from repository root with `C:/Python313/python.exe -X utf8 -B`.
Output: `artifacts/pyannote-lstm-input-root-patch-v3-20260922`.

Two preparation failures are preserved in the preceding namespaces. The first
omitted `.py` and `.proto` from known text formats, incorrectly counting two
newline-only differences. The second lacked Git's new-file mode headers and
failed `git apply --check`. Neither applied a patch or changed product source.
The explicit successor retains the same five-file content allowlist and passes
the application check. Its patch hash is
`c2821e18186083176923276001660d9a7b79d22bb154d996d526c1a591b8d829`.

Application is conditional on the completed, independently audited M22
application campaign passing every numerical, meeting, ownership, repeatability
and speed gate. A failed campaign prohibits applying the patch. An admitted
integration must verify the captured source identities again, apply exactly
these five files and compare all resulting source text to the qualified source.
Then build/test/package the normal root source on the AMD VM because workstation
free disk is below its unchanged local worker threshold. Expect all 3,163 Core
and 697 Data methods equivalent to the measured candidate, 3,432 backend passes
with the same 41 AMD skips, 343 tensor passes and independent actual NuGet
consumption. No push or package publication is part of this step.
