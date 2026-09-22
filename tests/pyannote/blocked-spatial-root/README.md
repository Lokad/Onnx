# Conditional integration of prepared convolution

`prepare_patch.py` produces a reviewed twelve-file patch from the qualified
normal candidate, including the corrected focused tests. It does not change
root product files. It verifies the unchanged selected source and both closed
candidate/package qualifications, then checks that the patch applies.
Known source/document text formats compare after newline normalization; binary
assets remain byte-exact. An initial pre-freeze check identified only the archive
versus Windows newline differences in documentation and fixture scripts.
Git application uses `--ignore-space-change` for the three existing CRLF context
files; all pre-change source bytes are pinned, and every post-change file must
equal the qualified source text. The default pre-freeze application check failed
on those contexts without changing source. The same retained patch may resume
preparation only while no manifest exists and its contents are exactly unchanged.

Only after the complete blocked-spatial-app-amd campaign has terminated and
passed its independent admission audit, run `run.py`, then terminal-only
`audit.py`. Use `C:/Python313/python.exe -X utf8 -B` from the repository root.
The integration tool checks every fixed numerical, meeting, repeatability and
speed gate, its recorded owners' termination, and the exact prepared patch
before applying any change. A failed campaign prohibits integration.

Normal root CLI/backend/tensor builds must preserve all 3,161 Core and 697 Data
method bodies and public declarations of the measured candidate. Full Windows
suites require 3,344 backend passes with 93 existing skips and 343 tensor passes.
An independently restored PackageReference consumer must import MNIST, execute
public operators and a prepared ConvRelu graph, preserve held outputs and inputs,
and validate two 1,056-value requests, 18,432 retained weight bytes and 8,384
requested scratch bytes. The actual root Core must be in the NuGet package;
Google.Protobuf 3.33.5 remains its only package dependency.

Build preflight requires 8 GiB available; numerical checks require 12 GiB.
Owned RSS remains below 8 GiB, each worker below 900 seconds, output below 1 GiB,
available memory at least 1 GiB, and disk free at least 20 GiB. CPU2 is inherited
before runtime startup; monitoring uses CPU0. All .NET commands disable the
terminal logger. The final audit freezes source snapshots and all resource
observations. Runtime equivalence does not create an additional timing score.

Patch output: `artifacts/pyannote-blocked-spatial-root-patch-20260922`.
Integration output: `artifacts/pyannote-blocked-spatial-root-20260922`.
Frozen or integration destinations are refused. There is no push or package publication.
