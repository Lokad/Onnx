# Normal composition of the single-panel convolution candidate

This isolated build combines the qualified direct-output kernel with reuse of
patches that already have the packed layout. It requires the completed, passing
single-panel AMD component screen and leaves the production worktree unchanged.

Run from repository root using `C:/Python313/python.exe -X utf8 -B`:
`prepare.py`, `audit.py prepare`, `qualify.py`, then `audit.py complete`, each
after the previous command and all of its owned processes have terminated.
The new directory is `artifacts/pyannote-single-panel-composition-20260922`.
Preparation archives root `521e41f6`; existing output is refused.

Normal project references build CLI/backend/tensors. Expected compiled changes
against accepted Core `e9c87932` / Data `85d166b5` are one existing caller and
five added methods, with 3,107 other Core / 697 Data methods and public surface
unchanged. The four generated arithmetic bodies must match the qualified
component. The corrected 400-case actual-caller program is reused unchanged in
normal and hardware-disabled modes, preserving exact non-NaN bits and recording
every differing NaN payload under the demonstrated baseline contract.

The wrapper keeps all admission, length and overlap checks. It rents and returns
packed storage only above 32 columns; narrower cases borrow the pinned readonly
patch. Focused tests require zero packed scratch for narrow cases and the exact
original requested bytes for wider cases. Expect 23 focused and 23 hardware-off
passes, 3,313 backend passes with 93 skips, and 343 tensor passes.

An independent NuGet consumer checks the packed Core identity, public operators,
ONNX import and ownership. It retains the existing 33,216-value tiled convolution
and adds a 10,240-value convolution with 32/8-column tiles, requiring exactly
327,680 initial scratch bytes and no subsequent packed rental. The package must
retain its sole Google.Protobuf 3.33.5 dependency.

Build/caller preflight requires 8 GiB available; suite/consumer preflight requires
10 GiB. Owned RSS stays below 8 GiB, workers below 900 seconds, output below
1 GiB, with at least 1 GiB available memory and 20 GiB free disk. CPU2 affinity
precedes CLR; the monitor uses CPU0. All .NET commands disable the terminal logger.

Complete native graphs, public requests, shared models, long meetings and fresh
AMD production/candidate/ORT timing remain required. Component admission does
not override the preceding application's failed 3% gate or establish parity.
