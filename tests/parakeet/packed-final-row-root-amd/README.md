# M78 root qualification preparation

This directory currently contains the prepared public-test adaptation. Root
integration and the full build/test/package adapter remain pending graph and
Pyannote application admission; no root product file has changed.

`public_tests.py` reads the qualified isolated `OwnedPackedWeightTests.cs` and
adds nine hardware guards using the repository's existing SkippableFact and
SkippableTheory conventions. Removing exactly those guards and attribute changes
must recover every original assertion, helper and inline case. It adds no
product changes and never edits the frozen isolated source.

`OwnedPackedWeightTests.cs.txt` is the generated integration input. Its SHA-256
is `6c17a21e90372683bd4d5682f6f0b39875674ed4f62a30d4f126e5fe5992ffcc`.
The receipt is
`artifacts/parakeet-packed-final-row-public-tests-20260925/prepared.json`.
Preparation is complete; do not rerun the exclusive generator.

Five local checks prove original-body preservation and reject assertion, guard
placement and inline-case drift. Case names come from the actual qualified
normal/AVX512-disabled/hardware-disabled TRX files: 40 ordinary public cases
and one hardware-unavailable case. Campaign runtime-identity tests are excluded
from the public suite. On this FMA-capable VM, require 40 new passes and one
new skip in both full-suite modes: backend 3,539/42 normal and 3,449/132 with
AVX512 disabled; tensor 394/0 in both. These are prospective expectations;
the generated public source has not yet been compiled in the full root suite.

After upstream admission, adapt the existing
`../slice-dense-conversion-root-amd` workflow. Integrate all exact measured M78
product files plus the guarded public test input and qualified slice tests.
Require every actual Core/Data method, flag and public declaration to match the
measured product, both full test censuses, unchanged warnings, and independent
NuGet consumption. Preserve existing package dependencies and original limits.
